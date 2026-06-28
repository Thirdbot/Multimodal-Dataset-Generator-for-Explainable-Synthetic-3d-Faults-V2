"""
Solely for llm response, for mechanic will be programmatic
"""
from operator import itemgetter
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from longtracer import check
from langchain_openai import ChatOpenAI


class AnswerBatchStructure(BaseModel):
    ANSWERS:list[str]

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
You are a geophysicist using only the Evidences as factual ground.
Never invent objects, values, causes, or events.
Do not mention graph, metadata, database, generated data, synthetic data, prompt, or verification.
Evidence tags: <object>...</object>, <nums>...</nums>, <center>...</center>, <bbox>...</bbox>.
Questions must be natural and must not copy tags.
Answers and reasoning MUST copy tagged spans exactly when using tagged evidence.
"""

answer_batch_generation_prompt = """
{master_prompt}

{format_instructions}

Output contract:
- Return only one valid JSON object.
- The first character must be {{ and the last character must be }}.
- Do not use markdown.
- Do not write text before or after the JSON object.
- Required shape: {{"ANSWERS":["one answer.","another answer."]}}

Generate candidate answers to one seismic interpretation question using only the provided evidences.

Rules:
- Generate up to {count} answers.
- One sentence per answer.
- Directly answer Question using only Evidences.
- Every factual word in the answer must be supported by Evidences.
- If Evidences do not answer the Question, return {{"ANSWERS":[]}}.
- Do not guess missing objects, counts, locations, regions, properties, fluids, or interpretations.
- Do not combine facts from different objects unless Evidences explicitly connect them.
- If using a tagged object/value/center/box, copy the full tagged span exactly.
- Keep the evidence format for reported facts: tagged object names stay tagged, tagged numbers stay tagged, tagged centers stay tagged, and tagged boxes stay tagged.
- Do not unwrap, rewrite, round, or paraphrase tagged spans.
- Do not replace a tagged value with an untagged value.
- Do not convert <nums>...</nums>, <center>...</center>, or <bbox>...</bbox> into plain text.
- Do not add unstated causes or interpretations.

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

Generate seismic interpretation questions from the provided evidences.

Rules:
- Generate up to {count} questions.
- Each item has QUESTION and RETRIEVAL_QUERY.
- QUESTION: natural GroundVQA-style visual question; no tags; no exact values; no answer leakage.
- QUESTION asks one answerable thing visible or described in Evidences.
- Use only object/property types present in Evidences.
- Ask about orientation only if Evidences mention tilt, dip, strike, angle, center, or bbox.
- If QUESTION compares or asks about multiple objects, QUESTION must name those objects clearly.
- RETRIEVAL_QUERY: 1 to 3 evidence-like sentence queries, one per line.
- RETRIEVAL_QUERY may use object names and tag words: object, nums, center, bbox.
- RETRIEVAL_QUERY must not be a keyword bag.

Good:
{{"QUESTIONS":[{{"QUESTION":"What geological feature is visible in this region?","RETRIEVAL_QUERY":"The section includes a visible object feature\nThe object occupies the area from bbox"}},{{"QUESTION":"How many visible episodes can be interpreted from the section?","RETRIEVAL_QUERY":"The layering shows nums onlap episodes"}},{{"QUESTION":"Where is the feature located?","RETRIEVAL_QUERY":"The feature sits near center\nThe feature occupies the area from bbox"}}]}}

Bad:
{{"QUESTIONS":[{{"QUESTION":"Does the object sit at x=43 and y=112?","RETRIEVAL_QUERY":"x=43 y=112"}},{{"QUESTION":"The onlap is yellow, right?","RETRIEVAL_QUERY":"onlap yellow"}},{{"QUESTION":"Where is the feature?","RETRIEVAL_QUERY":"feature center bbox object nums"}}]}}

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

Create a concise reasoning note that connects the Evidences to the Answer.

Rules:
- This is for audit and dataset explanation.
- Use one to three short sentences.
- Explain which evidence supports the Answer.
- If using a tagged object/value/center/box, copy the full tagged span exactly.
- Do not unwrap, rewrite, round, or paraphrase tagged spans.
- Do not add unstated causes or interpretations.

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

QuestionBatchPrompt = PromptTemplate(
    template=question_batch_generation_prompt,
    input_variables=["evidences","count"],
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
        self.max_tok = 256
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
