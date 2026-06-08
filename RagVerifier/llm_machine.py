"""
Solely for llm response, for mechanic will be programatic
"""
from operator import itemgetter
import json
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


hypothesis_qa_prompt = """
You are given verified graph evidence extracted from one generated seismic example.

  Your task is to answer the question with one atomic geological statement.

  Rules:
  - Return exactly one sentence.
  - The sentence must answer the question directly.
  - The sentence must contain exactly one claim.
  - Use only facts supported by the evidence.
  - Use the same object or property asked about in the question.
  - Do not combine multiple objects unless the evidence directly compares them.
  - Do not explain causes unless the evidence explicitly states the cause.
  - Do not infer geology that is not present in the evidence.
  - Avoid exact coordinates and decimal numbers unless the question asks for them.
  - Prefer simple geological wording.
  - Do not mention the graph, metadata, database, synthetic generation, or evidence source.
  - Do not say "as evidenced by", "according to", "based on", or "the evidence shows".
  - Do not mention uncertainty words like "maybe", "likely", or "appears" unless the evidence says that.
  - If the evidence is about absence, state only the absence claim.
  - If the question asks how many, answer only the count and object.
  - If the question asks whether something is present, answer only presence or absence.
  - Do not add examples, lists, reasons, locations, or secondary properties.

  Good atomic examples:
  - A fault is present.
  - No salt body is present.
  - The section contains one fault.
  - Fault 1 has measurable throw.
  - A gas closure is present.
  - An oil closure is present.
  - Sand-prone layering is present.
  - Onlap is present.
  - The section contains fan deposition.

  Bad examples:
  - The section is faulted and contains closures.
  - Faulting caused the closure.
  - The area is structurally complex because hydrocarbons are trapped.
  - The seismic image clearly shows salt and faults.
  - The model was generated with fault-only settings.
  - A fault is present as evidenced by eight faults with throw and positional data.
  - The section contains eight faults, indicating a total of eight fault intersections.

  Evidence:
  {evidence}

  Question:
  {question}

  Answer:
"""

question_generation_prompt = """
  You are given verified graph evidence extracted from one generated seismic example.

  Generate one question that can be answered by one atomic geological claim.

  Rules:
  - One question per line.
  - Each question must ask about exactly one feature or property.
  - Ask only about facts present in the evidence.
  - Do not ask for coordinates or decimal values unless they are central to the evidence.
  - Prefer questions useful for seismic interpretation.
  - Do not mention graph, metadata, database, synthetic generation, or evidence source.
  - Return only questions.
  - Do not write "Generated".
  - Do not write explanations.
  - Prefer questions that map to one property in the evidence.
  - Do not ask broad summary questions.

  Good questions:
  - Is a fault present?
  - Is salt present?
  - How many faults are present?
  - Is an oil closure present?
  - Is onlap present?
  - What kind of closure is present?
  - Does the section contain fan deposition?

  Bad questions:
  - What are all the structural features in this section?
  - What caused the faults and closures?
  - How complex is this section?

  Evidence:
  {evidence}

  Write one question:
"""

def clean_generated(text):
    # strip the word Generated
    text = str(text or "").strip()
    text = re.sub(r"^\s*Generated:\s*","",text,flags=re.I)
    return text.strip()

def parse_numbered_lines(text):
    text = clean_generated(text)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    items = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        line = re.sub(r"^(question|q)\s*\d*\s*[:\-]\s*", "", line, flags=re.I).strip()
        line = line.strip("-• ").strip()

        if line:
            items.append(line)

    return items


def parse_atomic_sentences(text):
    text = clean_generated(text)

    # Split by lines first, then sentence endings.
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line).strip()
        chunks.extend(re.split(r"(?<=[.!?])\s+", line))

    return [chunk.strip() for chunk in chunks if chunk.strip()]

class LLMMachine:
    def __init__(self):
        self.DEFAULT_VLLM_ENDPOINT = "http://localhost:8000/v1"
        self.temp = 0.1 # lower the better logic
        self.top_p = 0.95 # higher the better fluency
        self.max_tok = 512
        self.thinking_enable = False
        self.presence_penalty = 1 # -2,2 avoid repetition
        self.frequency_penalty = 0.2 # -2,2 more natural
        self.n = 1 # single response

        self.client = ChatOpenAI(base_url=self.DEFAULT_VLLM_ENDPOINT,
                                 api_key="None",
                                 model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                                 temperature=self.temp,
                                 frequency_penalty=self.frequency_penalty,
                                 presence_penalty=self.presence_penalty,
                                 top_p=self.top_p,
                                 max_tokens=self.max_tok,
                                 n=self.n)

        if self.thinking_enable:
            self.client = self.client.bind(
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "separate_reasoning": True,
                }
            )

    def format_docs(self,docs):
        return "\n".join(doc.page_content for doc in docs)


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

    def question_generation(self):
        q_query_engine = (
                {
                    "evidence":itemgetter("evidence"),
                } | ChatPromptTemplate.from_template(question_generation_prompt) | self.client | StrOutputParser()
        )
        return  q_query_engine

    def answer_generation(self):
        q_answer_engine = (
            {
                "evidence":itemgetter("evidence"),
                "question":itemgetter("question"),
            } | ChatPromptTemplate.from_template(hypothesis_qa_prompt) | self.client | StrOutputParser()
        )
        return q_answer_engine

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
        "evidence":example_evidence,
    })

    question_lists = parse_numbered_lines(response)
    print(question_lists)

    response = llm_machine.answer_generation().invoke({
        "evidence":example_evidence,
        "question":question_lists[0],
    })
    answered_lists = parse_atomic_sentences(response)
    print(answered_lists)
