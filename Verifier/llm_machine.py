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
You are a geophysicist reading evidence statements from a seismic interpretation workflow.
Use only the Evidences input as factual ground.
Write naturally, as a scientist explaining what the evidence says.
You may vary wording and style, but do not invent objects, values, causes, or geological events.
Do not mention graph, metadata, database, generated data, synthetic data, prompt, or verification.
Evidence sentences may contain markup:
- <object>...</object> marks the geological object being described.
- <nums>...</nums> marks numeric counts or measured properties.
- <center>...</center> marks a visible 2D object center.
- <bbox>...</bbox> marks a visible 2D region box.
Treat these tags as evidence annotations.
Questions must sound natural and must not copy the tags.
Answers and reasoning may preserve the tags when they report a tagged object, value, center, or region from evidence.
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
- Each answer should be one natural sentence.
- Each answer must directly answer the same Question.
- Every answer must be supported by Evidences.
- Answers may vary in wording, but each one must stay faithful to the same evidence.
- Use the same objects, quantities, properties, regions, or colors stated in Evidences.
- If an answer reports an object marked with <object>, keep the tag around that exact object name.
- If an answer reports a value or region marked with <nums>, <center>, or <bbox>, keep the tag around that exact value.
- Do not remove evidence tags from objects, values, centers, or regions that appear in the answer.
- Do not invent facts outside Evidences.
- Do not add causes or interpretations that are not stated in Evidences.

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
- Required shape: {{"QUESTIONS":[{{"QUESTION":"natural visual question?","RETRIEVAL_QUERY":"compact evidence lookup terms"}}]}}

Generate seismic interpretation questions from the provided evidences.

Rules:
- Generate up to {count} questions.
- Each item must include QUESTION and RETRIEVAL_QUERY.
- QUESTION is for a user looking at seismic image(s). It can be broad or concise, but must be answerable from Evidences.
- RETRIEVAL_QUERY is for evidence lookup. It should be compact, direct, and use object names, property names, and tagged evidence terms.
- RETRIEVAL_QUERY may include exact object names, property words, and tag words such as object, nums, center, bbox.
- RETRIEVAL_QUERY should not be conversational.
- Simulate GroundVQA-style curiosity: ask about the visible object, region, count, location, class, mask, or interpreted feature.
- QUESTION must not sound like it already knows the answer.
- If Evidences only mention onlap, ask only about onlap.
- If Evidences only mention salt, ask only about salt.
- If Evidences only mention a highlighted color or region, ask about that visual evidence.
- Do not ask about faults, closures, salt, onlap, fans, fluids, colors, or locations unless those words or facts appear in Evidences.
- Each question should ask about one clear answerable target.
- Every question must be directly answerable from Evidences.
- Questions are prompts for a user who has not seen the evidence values.
- The question must ask what the value/location/class/count is, not reveal it.
- Prefer open visual question forms such as "What is visible...", "Where is...", "Which feature...", "How many...", "What kind of...", "Does the image show...".
- Questions must be unique.
- If there are fewer unique evidence facts than {count}, return fewer questions.
- Use natural geological wording.
- Ask what the interpreter sees in the evidence statements.
- Convert evidence facts into image-facing questions.
- It is okay to ask about counts, object attributes, intersections, locations, colors, regions, and visible masks only when those facts are in Evidences.
- Do not include the answer inside QUESTION.
- Do not copy exact evidence values into the question.
- Do not put coordinates, bounding boxes, throw values, percentages, fluid names, colors, or counts in the question.
- Never write x=, y=, x_min, y_min, x_max, y_max, bbox, coordinate values, or "from x ... to ..." in QUESTION.
- Never copy <object>, <nums>, <center>, or <bbox> into QUESTION.
- Never ask a question that already contains the location, area, amount, or property value.
- Ask for the missing value, not with the value.
- Do not copy object types from examples; use only object types present in Evidences.
- Do not ask cause questions unless the cause is explicitly stated in Evidences.
- Make RETRIEVAL_QUERY match the expected evidence sentences, for example "Fault 1 object throw nums", "Closure 1 object bbox center fluid", or "onlap count nums".
- If QUESTION asks where something is, RETRIEVAL_QUERY must include center or bbox.
- If QUESTION asks how many, RETRIEVAL_QUERY must include count or nums.
- If QUESTION asks what type/class/fluid/property, RETRIEVAL_QUERY must include that property word.

Good:
{{"QUESTIONS":[{{"QUESTION":"What geological feature is visible in the highlighted region?","RETRIEVAL_QUERY":"highlighted region object bbox"}},{{"QUESTION":"How many visible episodes can be interpreted from the section?","RETRIEVAL_QUERY":"onlap episodes count nums"}},{{"QUESTION":"Where is the highlighted feature located?","RETRIEVAL_QUERY":"highlighted feature center bbox"}}]}}

Bad:
{{"QUESTIONS":[{{"QUESTION":"Does the object sit at x=43 and y=112?","RETRIEVAL_QUERY":"x=43 y=112"}},{{"QUESTION":"The onlap is yellow, right?","RETRIEVAL_QUERY":"onlap yellow"}}]}}

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
- Use two to four natural sentences.
- Mention the evidence facts that support the answer.
- If reasoning cites an object marked with <object>, keep the tag around that exact object name.
- If reasoning cites a value or region marked with <nums>, <center>, or <bbox>, keep the tag around that exact value.
- Do not remove evidence tags from objects, values, centers, or regions that appear in the reasoning.
- Do not add unstated causes, processes, or interpretations.
- Keep the conclusion aligned with the Answer.

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
    "Use the visible geological features, masks, overlays, and highlighted regions "
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
        self.max_tok = 128
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
            print(f"\t{a}: verdict = {checking.verdict} trust = {checking.trust_score}\n")
