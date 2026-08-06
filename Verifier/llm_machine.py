"""
Solely for llm response, for mechanic will be programmatic
"""
import os
from operator import itemgetter
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from longtracer import check
from langchain_openai import ChatOpenAI


class AnswerQueryPair(BaseModel):
    ANSWER:str
    RETRIEVAL_QUERY:str

class AnswerBatchStructure(BaseModel):
    ANSWERS:list[AnswerQueryPair]

class QuestionQueryPair(BaseModel):
    QUESTION:str
    RETRIEVAL_QUERY:str

class QuestionBatchStructure(BaseModel):
    QUESTIONS:list[QuestionQueryPair]


QuestionBatchParser = PydanticOutputParser(pydantic_object=QuestionBatchStructure)
AnswerBatchParser = PydanticOutputParser(pydantic_object=AnswerBatchStructure)

MASTER_PROMPT = """
You are a senior seismic interpreter (structural geologist) describing what a seismic section shows.
Ground every statement in the Evidences only. Never invent or contradict objects, counts, values, fluids, causes, or events, and add no interpretation the Evidences do not state.
Write in a natural, professional interpreter's voice -- plain geological language, not a data readout.
Never mention graphs, metadata, databases, models, synthetic or generated data, prompts, retrieval, or verification.
Write naturally and vary your phrasing -- reword sentences however reads best, and present a coordinate in whatever form fits (for example [x,y], (x,y), or "near x, y"). Keep every numeric VALUE and COORDINATE exactly as the Evidences give it (never invent, round, or change a number), and refer to each object by its coordinate the way the Evidences do. Always write a number as a DIGIT, never a word -- "2 faults", not "two faults". You may drop the <...> tags.
"""

answer_batch_generation_prompt = """
{master_prompt}

{format_instructions}

Output contract:
- Return only one valid JSON object.
- The first character must be {{ and the last character must be }}.
- Do not use markdown.
- Do not write text before or after the JSON object.
- Required shape: {{"ANSWERS":[{{"ANSWER":"one answer.","RETRIEVAL_QUERY":"evidence-like retrieval sentence"}}]}}

Task: answer the seismic interpretation Question using only the Evidences. Give up to {count} distinct candidate answers.

For each candidate:
- ANSWER: phrase it the way an interpreter would actually say it at the workstation. It may combine several facts, or more than one object, in one answer when the Question calls for it -- every fact you state must come from the Evidences, and connect two objects only where the Evidences connect them. If the Evidences give only a negative or section-level fact (no faults, none), state that directly and name no new object.
- When the Question asks to point out, outline, locate, or segment an object, identify that object clearly by its COORDINATE ("the fault at [57.5,289]") and state its grounded facts; refer to each object you localize by its coordinate, in a natural order. Write the ANSWER as PLAIN PROSE -- do NOT put any `<SEG>` token or other marker in it; the segmentation targets are attached downstream from the regions, so a `<SEG>` in the answer only corrupts the format.
- If the Question is COMPARATIVE or SUPERLATIVE ("the steeper fault", "the largest closure", "which has the greater throw"), read the Evidences like a geologist at the workstation and MAKE THE CALL with confidence: resolve it to a SPECIFIC object by its coordinate and STATE THE ACTUAL VALUES that justify the choice ("the fault at [57.5,289] is steeper, dipping 65 degrees versus 45 for the fault at [12,140]"). Comparing, ranking, or drawing a conclusion FROM the values the Evidences give is the interpreter's job -- it is a grounded inference, not invention, so do it directly rather than hedging or refusing; the verification checks your answer back against the Evidences, so a call that follows from the numbers will pass. Never assert a comparison without giving both compared values from the Evidences.
- Attribute a measured property to an object ONLY if the Evidences give that property FOR THAT object. Never say an object "dips", "has a throw", or "covers X percent" unless an Evidence line states it for that object -- e.g. onlap and salt have a coverage and a position but NO dip, so never call them "dipping"; a fault has dip and throw. If the Question asks "which is steeply dipping" or "which has the greatest throw", choose ONLY among objects that actually carry that value; if none do, say the section does not show it rather than pinning the property on the wrong object.
- RETRIEVAL_QUERY: copy, verbatim, the exact Evidence line(s) your ANSWER rests on -- one line per fact you used, each exactly as written in the Evidences. Do not reword, shorten, merge, or add to them; these lines are looked back up literally, so any paraphrase breaks the match. If the ANSWER leans on three facts, give the three Evidence lines behind them.
- If the Evidences do not answer the Question, return {{"ANSWERS":[]}}.

Evidences:
{evidences}

Question:
{question}

Return only JSON now:
"""

question_batch_generation_prompt = """
{master_prompt}

{format_instructions}

Output contract:
- Return only one valid JSON object.
- The first character must be {{ and the last character must be }}.
- Do not use markdown.
- Do not write text before or after the JSON object.
- Required shape: {{"QUESTIONS":[{{"QUESTION":"natural visual question?","RETRIEVAL_QUERY":"evidence-like retrieval sentence"}}]}}

Task: write up to {count} natural seismic-interpretation questions that the Evidences can answer.

For each item:
- QUESTION: a natural, GroundVQA-style question an interpreter would ask while reading the section -- no tags, no exact values, no answer given away. It may be simple (one property of one object) or compound (combine two properties, or compare or relate two named objects) -- but ONLY when the Evidences answer EVERY clause. Each clause must resolve to a specific value (dip, throw, area, count, fluid) or a named relation between coordinate-named objects; when you relate or compare two objects, NAME BOTH by their coordinate so both sides can be answered (both positions, both values). Never tack on a subjective or visual-only clause the Evidences cannot answer -- no "which is more prominent", "which stands out", "how it compares to the other structures", "most notable/striking", "visually" -- such a clause has no evidence to ground it, so the answer can only cover half. If the Evidences give only a negative or section-level fact (no faults, none), ask about the overall condition or the absence. Spread questions across these angles only where the Evidences allow, never forcing one: {facets}.
- Favor SEGMENTATION / LOCALIZATION questions: ask the reader to point out, outline, locate, or segment an object. **When the Evidences describe MORE THAN ONE object (they usually do), your batch must cover BOTH: (a) a DETAIL question about EACH object on its own (its dip / throw / area / fluid / position), AND (b) at least one question that COMPARES or RELATES two of them.** Both are answerable -- every object in the Evidences carries its full attributes -- so the comparison is one the evidence already supports, not a forced unanswerable clause; never let a multi-object seed produce ONLY single-object questions. For the comparison, pick something the two objects actually SHARE: a magnitude when they are the SAME type (two faults by dip or throw, two closures by area) -- steeper/gentler, more-displaced, larger/smaller, deeper/shallower -- or a SPATIAL RELATION by position (which is to the left/right/above the other) when they are DIFFERENT types (a fault and a closure share no magnitude, only position). Refer to an object by (1) its COORDINATE ("the fault at [x,y]"); (2) its SPATIAL RELATION to another ("the closure to the left of the fault at [a,b]"); or (3) a COMPARATIVE / SUPERLATIVE ATTRIBUTE the Evidences give the values for. NAME BOTH objects by their coordinate in any comparison. Examples: "segment the fault at [x,y]"; "which of the two faults dips more steeply -- [a,b] or [c,d]?"; "outline the closure to the left of the fault at [x,y]". A comparison or superlative over a SINGLE object is pointless -- only compare when the Evidences hold two or more of that thing. Only name objects the Evidences actually describe.
- Ask about orientation or geometry only if the Evidences mention dip, tilt, angle, center, or extent. In a compound or multi-object question, name each object clearly by its coordinate.
- RETRIEVAL_QUERY: copy, verbatim, the exact Evidence line(s) the Question rests on -- one line per fact the Question depends on, each exactly as written in the Evidences. Do not reword, shorten, or merge them; these lines are looked back up literally. A compound question lists every Evidence line it touches.
- Use only the object types, properties, and values present in the Evidences; introduce no new object, feature, or fluid.

Evidences:
{evidences}

Return only JSON now:
"""

multimodal_qa_instruction = (
    "Interpret the provided seismic images and answer the question. "
    "Use the visible geological features, masks, overlays, and regions "
    "when they are provided, and give a direct seismic interpretation answer."
)

DEFAULT_QUESTION_FACETS = "presence or absence, count, location, orientation, relationship"

QuestionBatchPrompt = PromptTemplate(
    template=question_batch_generation_prompt,
    input_variables=["evidences","count","facets"],
    partial_variables={
        "format_instructions":QuestionBatchParser.get_format_instructions(),
        "master_prompt": MASTER_PROMPT,
    }
)

AnswerBatchPrompt = PromptTemplate(
    template=answer_batch_generation_prompt,
    input_variables=["evidences","question","count"],
    partial_variables={
        "format_instructions":AnswerBatchParser.get_format_instructions(),
        "master_prompt": MASTER_PROMPT,
    }
)

def multimodal_dataset_instruction():
    return multimodal_qa_instruction

class LLMMachine:
    def __init__(self):
        self.DEFAULT_VLLM_ENDPOINT = "http://localhost:8000/v1"
        self.temp = 0.2 # lower the better logic
        self.top_p = 0.95 # higher the better fluency
        self.max_tok = 640  # room for a ~5-item batch (ANSWER + RETRIEVAL_QUERY each) without truncating the JSON
        self.presence_penalty = 1 # -2,2 avoid repetition
        self.frequency_penalty = 0.2 # -2,2 more natural
        self.n = 1 # single response
        self.attempt = 5

        self.client = ChatOpenAI(base_url=self.DEFAULT_VLLM_ENDPOINT,
                                 api_key="local",
                                 model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
                                 temperature=self.temp,
                                 frequency_penalty=self.frequency_penalty,
                                 presence_penalty=self.presence_penalty,
                                 top_p=self.top_p,
                                 max_tokens=self.max_tok,
                                 n=self.n)

        self.question_client = self.client.bind(
            temperature=0.6,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=1.2,
        )
        self.answer_client = self.client.bind(
            temperature=0.1,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )
    def question_batch_generation(self):
        q_query_engine = (
                {
                    "evidences":itemgetter("evidences"),
                    "count": lambda x: x.get("count", 5),
                    "facets": lambda x: x.get("facets", DEFAULT_QUESTION_FACETS),
                } | QuestionBatchPrompt | self.question_client | QuestionBatchParser
        ).with_retry(
        stop_after_attempt=self.attempt,
        retry_if_exception_type=(Exception,)
        )

        return q_query_engine

    def answer_batch_generation(self):
        q_answer_engine = (
            {
                "evidences":itemgetter("evidences"),
                "question":itemgetter("question"),
                "count": lambda x: x.get("count", 5),
            } | AnswerBatchPrompt | self.answer_client | AnswerBatchParser
        ).with_retry(
        stop_after_attempt=self.attempt,
        retry_if_exception_type=(Exception,)
        )

        return q_answer_engine

    def retrieve_many(self, retrieval):
        def _retrieve(query_text):
            queries = [line.strip() for line in str(query_text).splitlines() if line.strip()]
            docs = []
            seen = set()

            for query in queries:
                for doc in retrieval.invoke(query):
                    key = doc.page_content
                    if key in seen:
                        continue
                    seen.add(key)
                    docs.append(doc)

            return docs

        return _retrieve

if __name__ == "__main__":

    llm_machine = LLMMachine()
    example_evidence = """
    Fault 1 is present.
    Fault 1 has a throw of about 62.
    Fault 1 sits near x=43 and y=112 in the inline view.
    The section contains two faults.
  """

    batch = llm_machine.question_batch_generation().invoke({
        "evidences":example_evidence,
        "count": 3,
    })
    for q in batch.QUESTIONS:
        print(f"question: {q.QUESTION}\n")
        print(f"retrieval_query: {q.RETRIEVAL_QUERY}\n")
        answers = llm_machine.answer_batch_generation().invoke({
            "evidences":example_evidence,
            "question":q.QUESTION,
            "count": 3,
        })
        print(f"\tanswer: {answers.ANSWERS}\n")
        print("---")
        for a in answers.ANSWERS:
            checking = check(a,[example_evidence])
            print(f"\tanswer: {a}\n")
            print(f"\tverdict = {checking.verdict} trust = {checking.trust_score}\n")
