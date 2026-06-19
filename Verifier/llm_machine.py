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

class QuestionBatchStructure(BaseModel):
    QUESTIONS:list[str]


QuestionBatchParser = PydanticOutputParser(pydantic_object=QuestionBatchStructure)
AnswerBatchParser = PydanticOutputParser(pydantic_object=AnswerBatchStructure)

answer_batch_generation_prompt = """
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
- Each answer must be one natural sentence.
- Each answer must directly answer the same Question.
- Each answer must contain only one claim.
- Every answer must be supported by Evidences.
- Answers may be paraphrases, but each one must stay faithful to the same evidence.
- Ask the evidence for all it can support; do not ignore useful evidence when the question asks for it.
- Use the same objects, quantities, properties, regions, or colors stated in Evidences.
- Do not invent facts outside Evidences.
- Do not add causes or interpretations that are not stated in Evidences.
- Do not mention graph, metadata, database, generated data, or synthetic data.

Evidences:
{evidences}

Question:
{question}

Return only JSON now:
"""

question_batch_generation_prompt = """
{format_instructions}

Output contract:
- Return only one valid JSON object.
- The first character must be {{ and the last character must be }}.
- Do not use markdown.
- Do not write text before or after the JSON object.
- Required shape: {{"QUESTIONS":["question one?","question two?"]}}

Generate seismic interpretation questions from the provided evidences.

Rules:
- Generate up to {count} questions.
- Ask about everything that is directly supported by Evidences.
- Each question must ask about one clear answerable target.
- Every question must be directly answerable from Evidences.
- Questions must be unique.
- Do not generate two questions with the same meaning.
- Do not generate paraphrases of another question in the same list.
- Each question must target a different evidence fact.
- If there are fewer unique evidence facts than {count}, return fewer questions.
- Before returning, compare all questions and remove duplicates or near-duplicates.
- Cover the different evidence facts instead of repeating the same question.
- Use natural geological wording.
- It is okay to ask about counts, object attributes, intersections, locations, colors, regions, and visible masks when those facts are in Evidences.
- Do not invent objects or facts outside Evidences.
- Do not ask cause questions unless the cause is explicitly stated in Evidences.
- Do not mention graph, metadata, database, generated data, or synthetic data.

Good:
{{"QUESTIONS":["Is salt present?","How many faults are present?","Does Closure 1 contain gas?"]}}

Bad:
{{"QUESTIONS":["Is salt present?","Does the section contain salt?","Is there salt in the section?"]}}

Evidences:
{evidences}

Return only JSON now:
"""


multimodal_qa_instruction = (
    "Interpret the provided seismic images and answer the question. "
    "Use the visible geological features, masks, overlays, and highlighted regions "
    "when they are provided, and give a direct seismic interpretation answer."
)

QuestionBatchPrompt = PromptTemplate(
    template=question_batch_generation_prompt,
    input_variables=["evidences","count"],
    partial_variables={"format_instructions":QuestionBatchParser.get_format_instructions()}
)

AnswerBatchPrompt = PromptTemplate(
    template=answer_batch_generation_prompt,
    input_variables=["evidences","question","count"],
    partial_variables={"format_instructions":AnswerBatchParser.get_format_instructions()}
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
        print(f"question: {q}\n")
        answers = llm_machine.answer_batch_generation().invoke({
            "evidences":example_evidence,
            "question":q,
            "count": 3,
        })
        print(f"\tanswer: {answers.ANSWERS}\n")
        print("---")
        for a in answers.ANSWERS:
            checking = check(a,[example_evidence])
            print(f"\t{a}: verdict = {checking.verdict} trust = {checking.trust_score}\n")

