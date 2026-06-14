"""
Solely for llm response, for mechanic will be programmatic
"""
from operator import itemgetter
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI


class ReasonStructure(BaseModel):
    REASON:str

class AnswerStructure(BaseModel):
    ANSWER:str

class QuestionStructure(BaseModel):
    QUESTION:str


QuestionParser = PydanticOutputParser(pydantic_object=QuestionStructure)
ReasonParser = PydanticOutputParser(pydantic_object=ReasonStructure)
AnswerParser = PydanticOutputParser(pydantic_object=AnswerStructure)

reasoning_prompt = """
{format_instructions}

Output contract:
- Return only one valid JSON object.
- The first character must be {{ and the last character must be }}.
- Do not use markdown.
- Do not write REASON: labels.
- Do not write text before or after the JSON object.
- Required shape: {{"REASON":"detailed audit reason"}}

You write an audit reason explaining how the provided evidences relate to the question.
This reason is for debugging and trace inspection only. It will not be used as the verifier.

Rules:
- REASON can be two to four sentences.
- REASON must cite the important evidence facts in natural language.
- REASON must explain which evidence directly answers the question.
- REASON may mention distractor evidence when it could confuse the answer.
- REASON may state that the question is not answerable if the needed evidence is absent.
- REASON may use position, center, or bounding-box facts only as region-grounding context.
- Do not turn raw x/y coordinates or bounding boxes into the main answer logic unless the question explicitly asks for location.
- Prefer "the highlighted feature area" or "the visible feature region" instead of raw coordinate wording.
- Do not infer causes.
- Do not add geological interpretation not stated in Evidences.
- Do not mention graph, metadata, database, generated data, or synthetic data.
- If the question asks a numeric comparison, explain the exact values involved and whether the comparison is directly supported.
- If the question asks a count, use the count stated in Evidences.
- If the question asks about an object, use the same object.
- Do not invent evidence that is not listed.

Good:
Question: Does Closure 1 avoid onlap?
Evidences: Closure 1 avoids onlap. Closure 2 contains gas.
Output: {{"REASON":"The question asks only about Closure 1 and onlap. The relevant evidence directly states that Closure 1 avoids onlap. The Closure 2 fluid evidence is not needed for this answer."}}

Good:
Question: How many faults are present?
Evidences: The section shows 2 faults. Fault 1 has throw of about 12.2922.
Output: {{"REASON":"The count evidence states that the section shows 2 faults. The throw evidence describes Fault 1 but does not change the fault count."}}

Evidences:
{evidences}

Question:
{question}

Return only JSON now:
"""

hypothesis_qa_prompt = """
{format_instructions}

Output contract:
- Return only one valid JSON object.
- The first character must be {{ and the last character must be }}.
- Do not use markdown.
- Do not write ANSWER: or REASON: labels.
- Do not write text before or after the JSON object.
- Required shape: {{"ANSWER":"one sentence"}}

You answer seismic interpretation questions using only the provided evidences.

Evidence schema:
- Fault objects are named Fault 1, Fault 2, etc.
- Closure objects are named Closure 1, Closure 2, etc.
- "contains oil/gas/brine" means closure fluid type.
- "avoids fault/salt/onlap" means no intersection with that feature.
- "intersects fault/salt/onlap" means the object touches or crosses that feature.
- "throw", "tilt", "shear zone", and "gouge" are fault properties.
- "The section shows N faults" means the fault count.
- "Salt is present" means salt exists in the section.
- "Sand-prone intervals" and "fan deposition" describe depositional content.
- Position, center, or bounding-box facts are region metadata for grounding, not the main interpretation.

Rules:
- ANSWER must be one natural sentence.
- ANSWER must directly answer the question.
- ANSWER must contain only one claim.
- Use only facts stated in Evidences.
- Use the same object asked about in the Question.
- Do not add causes, interpretations, or extra properties.
- Do not mention graph, metadata, evidence, database, generated data, or synthetic data.
- Do not mention region ids, bounding boxes, or x/y coordinates unless the question explicitly asks for location.
- If location is explicitly asked, describe it qualitatively as the visible or highlighted feature area instead of listing raw coordinates.
- If the Question asks "Does Closure 1 avoid onlap?", answer "Closure 1 avoids onlap."
- If the Question asks "Does Closure 1 contain gas?", answer "Closure 1 contains gas."
- If the Question asks "How many faults are present?", answer "The section contains N faults."
- If a yes/no question is false, start with "No," and state the supported fact.
- Do not answer threshold questions unless the exact comparison is directly supported.
- Use Reason only as an audit hint; the final ANSWER must still be supported by Evidences.

Good:
Question: Does Closure 1 avoid onlap?
Output: {{"ANSWER":"Closure 1 avoids onlap."}}

Good:
Question: How many faults are present?
Output: {{"ANSWER":"The section contains one fault."}}

Good:
Question: Is the section composed of multiple faults?
Evidence says: The section shows 1 fault.
Output: {{"ANSWER":"No, the section contains one fault."}}

Bad:
{{"ANSWER":"Closure 1 avoids onlap because it formed away from structural growth."}}
Bad:
{{"ANSWER":"The graph shows Closure 1 avoids onlap."}}
Bad:
ANSWER: Yes, there is a fault with a shear zone wider than 0.

Evidences:
{evidences}

Reason:
{reason}

Question:
{question}

Return only JSON now:
"""

question_generation_prompt = """
{format_instructions}

Output contract:
- Return only one valid JSON object.
- The first character must be {{ and the last character must be }}.
- Do not use markdown.
- Do not write QUESTION: labels.
- Do not write text before or after the JSON object.
- Required shape: {{"QUESTION":"one question?"}}

You generate one seismic interpretation question from the provided evidences.

Evidence schema:
- Fault objects are named Fault 1, Fault 2, etc.
- Closure objects are named Closure 1, Closure 2, etc.
- "contains oil/gas/brine" means closure fluid type.
- "avoids fault/salt/onlap" means no intersection with that feature.
- "intersects fault/salt/onlap" means the object touches or crosses that feature.
- "throw", "tilt", "shear zone", and "gouge" are fault properties.
- "The section shows N faults" means the fault count.
- "Salt is present" means salt exists in the section.
- "Sand-prone intervals" and "fan deposition" describe depositional content.
- Position, center, or bounding-box facts are region metadata for image grounding.

Rules:
- QUESTION must be one natural question.
- QUESTION must ask about exactly one fact from Evidences.
- Use natural geological wording, not raw graph keys.
- Ask only questions that can be directly answered from Evidences.
- Prefer questions about visible or highlighted geological features, such as faults, closures, salt, onlap, or depositional content.
- Do not mention graph, metadata, evidence, database, generated data, or synthetic data.
- Do not mention region ids, bounding boxes, x/y coordinates, centers, or raw positions.
- Do not ask "where" questions unless the evidence gives only qualitative position.
- Do not ask broad summary questions.
- Do not ask cause questions.
- Do not ask threshold/comparison questions using greater than, less than, wider than, or higher than.
- Do not ask numeric comparison questions even when decimal values are present.
- Do not use words like above, below, at least, more than, under, exceed, larger, smaller, wider, greater, or less.
- Do not ask whether something is "multiple"; ask exact counts instead.
- Use "Does Closure 1 avoid onlap?", not "Is closure 1 avoid onlap?"
- Use "Does Closure 1 contain gas?", not "Is closure 1 fluid gas?"

Good:
Evidence: Closure 1 avoids onlap.
Output: {{"QUESTION":"Does Closure 1 avoid onlap?"}}

Good:
Evidence: Closure 2 contains gas.
Output: {{"QUESTION":"Does Closure 2 contain gas?"}}

Good:
Evidence: The section shows 1 fault.
Output: {{"QUESTION":"How many faults are present?"}}

Bad:
{{"QUESTION":"Is closure 1 avoid onlap?"}}
Bad:
{{"QUESTION":"Is closure 1 fluid gas?"}}
Bad:
{{"QUESTION":"Is there a fault with a throw greater than 12.2922?"}}
Bad:
QUESTION: What are all the structural features in this section?

Evidences:
{evidences}

Return only JSON now:
"""


multimodal_qa_instruction = (
    "Interpret the provided seismic images and answer the question. "
    "Use the visible geological features, masks, overlays, and highlighted regions "
    "when they are provided, and give a direct seismic interpretation answer."
)

QuestionPrompt = PromptTemplate(
    template=question_generation_prompt,
    input_variables=["evidences"],
    partial_variables={"format_instructions":QuestionParser.get_format_instructions()}
)

ReasonPrompt = PromptTemplate(
    template=reasoning_prompt,
    input_variables=["evidences","question"],
    partial_variables={"format_instructions":ReasonParser.get_format_instructions()}
)

AnswerPrompt = PromptTemplate(
    template=hypothesis_qa_prompt,
    input_variables=["evidences","question","reason"],
    partial_variables={"format_instructions":AnswerParser.get_format_instructions()}
)


def multimodal_dataset_instruction():
    return multimodal_qa_instruction


class LLMMachine:
    def __init__(self):
        self.DEFAULT_VLLM_ENDPOINT = "http://localhost:8000/v1"
        self.temp = 0.2 # lower the better logic
        self.top_p = 0.95 # higher the better fluency
        self.max_tok = 256
        self.presence_penalty = 1 # -2,2 avoid repetition
        self.frequency_penalty = 0.2 # -2,2 more natural
        self.n = 1 # single response
        self.attempt = 5

        self.client = ChatOpenAI(base_url=self.DEFAULT_VLLM_ENDPOINT,
                                 api_key="None",
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
        self.reason_client = self.client.bind(
            temperature=0.2,
            top_p=0.95,
            frequency_penalty=0.2,
            presence_penalty=0.8,
        )
        self.answer_client = self.client.bind(
            temperature=0.1,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )


    def question_generation(self):
        q_query_engine = (
                {
                    "evidences":itemgetter("evidences"),
                } |QuestionPrompt | self.question_client | QuestionParser
        ).with_retry(
        stop_after_attempt=self.attempt,
        retry_if_exception_type=(Exception,)
        )

        return  q_query_engine

    def reason_generation(self):
        q_reason_engine = (
            {
                "evidences":itemgetter("evidences"),
                "question":itemgetter("question"),
            } | ReasonPrompt | self.reason_client | ReasonParser
        ).with_retry(
        stop_after_attempt=self.attempt,
        retry_if_exception_type=(Exception,)
        )

        return q_reason_engine

    def answer_generation(self):
        q_answer_engine = (
            {
                "evidences":itemgetter("evidences"),
                "question":itemgetter("question"),
                "reason": lambda x: x.get("reason", ""),
            } | AnswerPrompt | self.answer_client | AnswerParser
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
    Fault 1 has measurable throw.
    Fault 1 has a throw of about sixty two.
    Fault 1 sits near x=forty three and y=one hundred twelve in the inline view.
    The section contains two faults.
    Fault voxels are present.
  """

    response = llm_machine.question_generation().invoke({
        "evidences":example_evidence,
    })
    print(response.QUESTION)
    reason = llm_machine.reason_generation().invoke({
        "evidences":example_evidence,
        "question":response.QUESTION,
    })
    print(reason.REASON)
    response = llm_machine.answer_generation().invoke({
        "evidences":example_evidence,
        "question":response.QUESTION,
        "reason":reason.REASON,
    })
    print(response.ANSWER)
