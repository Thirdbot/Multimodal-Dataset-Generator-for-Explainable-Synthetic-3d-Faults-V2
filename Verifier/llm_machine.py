"""
Solely for llm response, for mechanic will be programmatic
"""
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

class ReasonStructure(BaseModel):
    REASON:str


QuestionBatchParser = PydanticOutputParser(pydantic_object=QuestionBatchStructure)
AnswerBatchParser = PydanticOutputParser(pydantic_object=AnswerBatchStructure)
ReasonParser = PydanticOutputParser(pydantic_object=ReasonStructure)

MASTER_PROMPT = """
You are a senior seismic interpreter (structural geologist) describing what a seismic section shows.
Ground every statement in the Evidences only. Never invent or contradict objects, counts, values, fluids, causes, or events, and add no interpretation the Evidences do not state.
Write in a natural, professional interpreter's voice -- plain geological language, not a data readout.
Never mention graphs, metadata, databases, models, synthetic or generated data, prompts, retrieval, or verification.
Copy any tagged span (<object>...</object>, <nums>...</nums>, <center>...</center>, <bbox>...</bbox>) exactly when you use it; do not unwrap, round, or reword it. Questions never contain tags.
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
- ANSWER: one concise sentence, phrased the way an interpreter would actually say it at the workstation. Use only the objects, properties, fluids, and values found in the Evidences, and connect two objects only where the Evidences connect them. If the Evidences give only a negative or section-level fact (no faults, none), state that directly and name no new object.
- RETRIEVAL_QUERY: the plain facts behind the answer, one short evidence-like sentence per line, each naming its object exactly as written in the Evidences (this is used to look the facts back up, so keep it literal, not a keyword bag).
- Copy any tagged span (<object>, <nums>, <center>, <bbox>) exactly in the ANSWER; never turn a tagged value into plain text.
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
- QUESTION: a natural, GroundVQA-style question an interpreter would ask while reading the section -- no tags, no exact values, no answer given away. Ask one thing the Evidences actually support. If the Evidences give only a negative or section-level fact (no faults, none), ask about the overall condition or the absence. Spread questions across these angles only where the Evidences allow, never forcing one: {facets}.
- Ask about orientation or geometry only if the Evidences mention dip, tilt, angle, center, or extent. If a question involves more than one object, name them clearly.
- RETRIEVAL_QUERY: the fact(s) the question rests on, one short evidence-like sentence per line, each naming its object exactly as written in the Evidences (keep it literal, not a keyword bag).
- Use only the object types, properties, and values present in the Evidences; introduce no new object, feature, or fluid.

Evidences:
{evidences}

Return only JSON now:
"""

reason_generation_prompt = """
{master_prompt}

{format_instructions}

Output contract:
- Return only one valid JSON object.
- The first character must be {{ and the last character must be }}.
- Do not use markdown.
- Do not write text before or after the JSON object.
- Required shape: {{"REASON":"short evidence-guided reasoning."}}

Task: in two or three short steps, explain like an interpreter why the Answer follows from the Evidences. The Answer is already correct and verified -- justify it, never re-answer it, and never say the Evidences are missing or empty.

- Walk from what the Evidence states, to what it implies about the section, to why the Answer follows, naming the specific evidence fact it rests on.
- If the Evidences state a negative or section-level fact (no faults, none), explain that the Answer reflects that stated absence.
- Copy any tagged span exactly; add no cause or interpretation the Evidences do not state.

Evidences:
{evidences}

Question:
{question}

Answer:
{answer}

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

ReasonPrompt = PromptTemplate(
    template=reason_generation_prompt,
    input_variables=["evidences", "question", "answer"],
    partial_variables={
        "format_instructions":ReasonParser.get_format_instructions(),
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
                                 model="Qwen/Qwen2.5-1.5B-Instruct",
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
        self.reason_client = self.client.bind(
            temperature=0.2,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.4,
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

    def reason_generation(self):
        reason_engine = (
            {
                "evidences":itemgetter("evidences"),
                "question":itemgetter("question"),
                "answer":itemgetter("answer"),
            } | ReasonPrompt | self.reason_client | ReasonParser
        ).with_retry(
        stop_after_attempt=self.attempt,
        retry_if_exception_type=(Exception,)
        )

        return reason_engine

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
