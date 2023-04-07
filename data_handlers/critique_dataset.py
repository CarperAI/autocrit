from datasets import Dataset
import jsonlines

def extract_data(file_path):

    critique_data_list = list()
    refinement_data_list = list()
    crit_ref_data_list = list()
    crit_rm_data_list = list()
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            for item in obj['data']['questions']:
                if item['question'] == '':
                    continue
                if obj['data']['passage']['title'] is not None:
                    start = f"{obj['data']['passage']['title']}\n{obj['data']['passage']['text']}\n\n"
                else:
                    start = f"{obj['data']['passage']['text']}\n\n"
                for answer in item['answers']:
                    if answer['feedback'].get('skip', False):
                        continue
                    if answer['answer'] == '':
                        continue
                    if (answer['feedback']['rating'] >= 7) and (len(answer['feedback']['critiques']) == 0):
                        critique_data_list.append(
                            {
                                "id": obj['id'],
                                "source_id": obj['source_id'],
                                "split": obj['split'],
                                "time": obj['time'],
                                "labeler": obj['labeler'],
                                "is_topic_based_summarization": obj["is_topic_based_summarization"],
                                "category": "N/A",
                                "severity": -1,
                                "text_quotes": [],
                                "response_quotes": [],
                                "prompt": f"{start}"
                                          f"Question: {item['question']}\n\n"
                                          f"Answer: {answer['answer']}",
                                "response": f"Critiqueable: No"
                            }
                        )

                    for critique in answer['feedback']['critiques']:
                        critique_data_list.append(
                            {
                                "id": obj['id'],
                                "source_id": obj['source_id'],
                                "split": obj['split'],
                                "time": obj['time'],
                                "labeler": obj['labeler'],
                                "is_topic_based_summarization": obj["is_topic_based_summarization"],
                                "category": critique.get('category', "N/A"),
                                "severity": critique.get('severity', -1),
                                "text_quotes": critique.get('text_quotes', []),
                                "response_quotes": critique.get('response_quotes', []),
                                "prompt": f"{start}"
                                          f"Question: {item['question']}\n\n"
                                          f"Answer: {answer['answer']}",
                                "response": f"Critiqueable: Yes\n\nCritique: {critique['text']}"
                            }
                        )
                        refinement_data_list.append(
                            {
                                "id": obj['id'],
                                "source_id": obj['source_id'],
                                "split": obj['split'],
                                "time": obj['time'],
                                "labeler": obj['labeler'],
                                "is_topic_based_summarization": obj["is_topic_based_summarization"],
                                "category": critique.get('category', "N/A"),
                                "severity": critique.get('severity', -1),
                                "text_quotes": critique.get('text_quotes', []),
                                "response_quotes": critique.get('response_quotes', []),
                                "prompt": f"{start}"
                                          f"Question: {item['question']}\n\n"
                                          f"Answer: {answer['answer']}\n\n"
                                          f"Critiqueable: Yes\n\nCritique: {critique['text']}",
                                "response": f"Refinement: {critique['refinement']}"
                            }
                        )
                        crit_ref_data_list.append(
                            {
                                "id": obj['id'],
                                "source_id": obj['source_id'],
                                "split": obj['split'],
                                "time": obj['time'],
                                "labeler": obj['labeler'],
                                "is_topic_based_summarization": obj["is_topic_based_summarization"],
                                "category": critique.get('category', "N/A"),
                                "severity": critique.get('severity', -1),
                                "text_quotes": critique.get('text_quotes', []),
                                "response_quotes": critique.get('response_quotes', []),
                                "prompt": f"{start}"
                                          f"Question: {item['question']}\n\n"
                                          f"Answer: {answer['answer']}",
                                "response": f"Critiqueable: Yes\n\nCritique: {critique['text']}\n\n"
                                            f"Refinement: {critique['refinement']}"
                            }
                        )
                crit_rm_data_list.append(
                    {
                        "id": obj['id'],
                        "source_id": obj['source_id'],
                        "split": obj['split'],
                        "time": obj['time'],
                        "labeler": obj['labeler'],
                        "is_topic_based_summarization": obj["is_topic_based_summarization"],
                        "prompt": f"{start}"
                                  f"Question: {item['question']}",
                        "answers": [item['answers'][j]
                                    for j in range(len(item['answers']))],
                        "rankings": item['rankings']
                    }
                )
    return critique_data_list, refinement_data_list, crit_ref_data_list, crit_rm_data_list
                

if __name__ == '__main__':
    train_crit, train_ref, train_crit_ref, train_rm_data = extract_data(r"train.jsonl(1)\train.jsonl")
    test_crit, test_ref, test_crit_ref, test_rm_data = extract_data(r"test.jsonl(1)\test.jsonl")
    Dataset.from_list(test_crit).push_to_hub("self-critiquing-critique-test")
    Dataset.from_list(test_ref).push_to_hub("self-critiquing-refine-test")
    Dataset.from_list(test_crit_ref).push_to_hub("self-critiquing-critique-and-refine-test")
    Dataset.from_list(test_rm_data).push_to_hub("self-critiquing-critique-answer-ranking-test")
    Dataset.from_list(train_crit).push_to_hub("self-critiquing-critique-train")
    Dataset.from_list(train_ref).push_to_hub("self-critiquing-refine-train")
    Dataset.from_list(train_crit_ref).push_to_hub("self-critiquing-critique-and-refine-train")
    Dataset.from_list(train_rm_data).push_to_hub("self-critiquing-critique-answer-ranking-train")
    