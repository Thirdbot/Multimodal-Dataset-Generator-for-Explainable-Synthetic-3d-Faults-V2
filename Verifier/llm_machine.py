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
- Required shape: {{"QUESTIONS":["question one?","question two?"]}}

Generate seismic interpretation questions from the provided evidences.

Rules:
- Generate up to {count} questions.
- Ask only about what appears in Evidences.
- Write questions as if the user will ask them while looking at seismic images, not while reading the evidence text.
- Simulate geological curiosity: ask what the image appears to show, where a feature is, what kind of feature it is, or what interpretation is supported.
- The question should sound like a natural visual interpretation question for a VLM.
- The question must not sound like it already knows the answer.
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
- Do not include the answer inside the question.
- Do not copy exact evidence values into the question.
- Do not put coordinates, bounding boxes, throw values, percentages, fluid names, colors, or counts in the question.
- Never write x=, y=, x_min, y_min, x_max, y_max, bbox, coordinate values, or "from x ... to ..." in a question.
- Never ask a question that already contains the location, area, amount, or property value.
- Ask for the missing value, not with the value.
- Do not copy object types from examples; use only object types present in Evidences.
- Do not ask cause questions unless the cause is explicitly stated in Evidences.

Good:
{{"QUESTIONS":["What geological feature is visible in the highlighted region?","How many visible episodes can be interpreted from the section?","Where is the highlighted feature located?","What color marks the interpreted feature?","What property can be interpreted for the highlighted object?","Does the image show the interpreted feature?"]}}

Bad:
{{"QUESTIONS":["How many faults are present?","What fluid does Closure 1 contain?","Where is Fault 1 located?","Does the object sit at x=43 and y=112?","Is the highlighted object red?","The onlap is yellow, right?"]}}

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
        self.DEFAULT_VLLM_ENDPOINT = "https://pd-third-qwen2-7b-i-c04894ffc6824922a1cc83d17c3b8bcc.nebius-eu-north.saturnenterprise.io/v1"
        self.temp = 0.2 # lower the better logic
        self.top_p = 0.95 # higher the better fluency
        self.max_tok = 1024
        self.presence_penalty = 1 # -2,2 avoid repetition
        self.frequency_penalty = 0.2 # -2,2 more natural
        self.n = 1 # single response
        self.attempt = 5

        self.client = ChatOpenAI(base_url=self.DEFAULT_VLLM_ENDPOINT,
                                 api_key=None,
                                 model="RedHatAI/Qwen2-7B-Instruct-quantized.w4a16",
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
