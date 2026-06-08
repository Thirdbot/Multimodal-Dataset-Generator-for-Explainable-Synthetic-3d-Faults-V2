from longtracer import LongTracer,check

from rag_verifier import serialize_docs, best_doc_score, score_qa_evidence
from llm_machine import parse_numbered_lines, LLMMachine
from create_rag import Rag

# create rag for each rag run pipeline

if __name__ == "__main__":
    rag = Rag(embedding_model="all-MiniLM-L6-v2")
    llm_machine = LLMMachine()

    LongTracer.init(verbose=True)

    number_of_question = 5
    number_of_candidate = 5

    for graph in rag.get_graph():  # 1 graph
        # retrieving rag
        vector_store, edges = rag.mapping_graph_rag(graph)  # Document objects
        graph_retrieval = rag.graph_retrieval(vector_store, edges)
        # instrument_langchain(graph_retrieval) # tracking
        all_sentences = rag.get_all(graph)  # all evidences

        # pass in to generating question
        evidences = all_sentences

        qa_store = []
        for _ in range(number_of_question):
            qa_template = {
                "question": "",
                "q_evidence": [],
                "answer": "",
                "a_evidence": []
            }
            answer_store = []
            # generate question
            response = llm_machine.question_generation().invoke({
                "evidence": evidences,
            })

            question_lists = parse_numbered_lines(response)

            if not question_lists:
                print("no question generated")
                continue

            one_question = question_lists[0]  # 1 question

            question_retrieve_evidence = graph_retrieval.invoke(one_question)  # evidences retrieved by question
            string_question_retrieve_evidence = rag.format_docs(question_retrieve_evidence)  # turn into string

            if best_doc_score(question_retrieve_evidence) < 0.7:
                print("no question generated")
                continue

            # add question
            qa_template["question"] = one_question
            qa_template["q_evidence"] = serialize_docs(question_retrieve_evidence)

            print(one_question)

            # all candidate hypotheses from all evidences and question
            for _ in range(number_of_candidate):
                answer_template = {
                    "answer": "",
                    "evidence": [],
                    "score": 0.0,
                }

                response = llm_machine.answer_generation().invoke({
                    "evidence": evidences,
                    "question": one_question,
                })

                answered_lists = parse_numbered_lines(response)

                if not answered_lists:
                    print("no answer generated")
                    continue

                one_answer = answered_lists[0]  # 1 answer hypothesis

                answer_retrieve_evidence = graph_retrieval.invoke(one_answer)  # evidences retrieved by answer

                qa_alignment_score = score_qa_evidence(question_retrieve_evidence, answer_retrieve_evidence)
                # Case that the answer is not answering to question (a_evidence must match q_evidence)
                if qa_alignment_score < 1.0:
                    print("wrong answer from question")
                    continue

                print(one_answer)

                string_answer_retrieve_evidence = rag.format_docs(answer_retrieve_evidence)  # turn into string

                # check premise(evidences) and hypothesis(one_answer)
                verify = check(one_answer, [evidences])  # check answer by all evidences
                trust_score = float(verify.trust_score)
                verdict = verify.verdict

                answer_template["answer"] = one_answer
                answer_template["evidence"] = serialize_docs(answer_retrieve_evidence)
                answer_template["score"] = trust_score

                answer_store.append(answer_template)

            answer_store = sorted(answer_store, key=lambda x: x["score"], reverse=True)  # select the best answer

            if not answer_store:
                print("no answer generated. skip question")
                continue

            qa_template["answer"] = answer_store[0]['answer']  # no, answer then skip question
            qa_template["a_evidence"] = answer_store[0]['evidence']
            qa_store.append(qa_template)

        print(qa_store)
        break  # 1 graph
