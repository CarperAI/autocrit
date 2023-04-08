from datasets import Dataset, splits, DatasetDict
import jsonlines


def extract_helpful_and_rm_data(data_path):
    data_list = list()
    rm_data_list = list()
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            for item in obj['data']['questions']:
                if item['question'] == '':
                    continue
                if obj['data']['passage']['title'] is not None:
                    start = f"{obj['data']['passage']['title']}\n{obj['data']['passage']['text']}\n\n"
                else:
                    start = f"{obj['data']['passage']['text']}\n\n"
                if item['is_critiqueable'] is not None:
                    if not item['is_critiqueable']:
                        continue
                for i in range(len(item['answers'])):
                    if item['has_better_critique']:
                        rm_data_list.append(
                            {
                                "id": obj['id'],
                                "source_id": obj['source_id'],
                                "split": obj['split'],
                                "time": obj['time'],
                                "labeler": obj['labeler'],
                                "is_topic_based_summarization": obj["is_topic_based_summarization"],
                                "prompt": f"{start}"
                                          f"Question: {item['question']}\n\n"
                                          f"Answer: {item['answers'][0]['answer']}\n\n"
                                          f"Critiqueable: Yes\n\n"
                                          f"Critique: {item['new_critique']['text']}",
                                "helpful": True
                            }
                        )
                        data_list.append(
                            {
                                "id": obj['id'],
                                "source_id": obj['source_id'],
                                "split": obj['split'],
                                "time": obj['time'],
                                "labeler": obj['labeler'],
                                "is_topic_based_summarization": obj["is_topic_based_summarization"],
                                "prompt": f"{start}"
                                          f"Question: {item['question']}\n\n"
                                          f"Answer: {item['answers'][0]['answer']}",
                                "response": f"Critiqueable: Yes\n\n"
                                            f"Critique: {item['new_critique']['text']}"
                            }
                        )
                    for feedback in item['answers'][i]['critiques']:
                        rm_data_list.append(
                            {
                                "id": obj['id'],
                                "source_id": obj['source_id'],
                                "split": obj['split'],
                                "time": obj['time'],
                                "labeler": obj['labeler'],
                                "is_topic_based_summarization": obj["is_topic_based_summarization"],
                                "prompt": f"{start}"
                                          f"Question: {item['question']}\n\n"
                                          f"Answer: {item['answers'][i]['answer']}\n\n"
                                          f"Critiqueable: Yes\n\n"
                                          f"Critique: {feedback['critique']}",
                                "helpful": feedback['feedback']['helpful']
                            }
                        )
                        if not feedback['feedback']['helpful']:
                            continue
                        if item['has_better_critique']:
                            continue
                        data_list.append(
                            {
                                "id": obj['id'],
                                "source_id": obj['source_id'],
                                "split": obj['split'],
                                "time": obj['time'],
                                "labeler": obj['labeler'],
                                "is_topic_based_summarization": obj["is_topic_based_summarization"],
                                "prompt": f"{start}"
                                          f"Question: {item['question']}\n\n"
                                          f"Answer: {item['answers'][i]['answer']}",
                                "response": f"Critiqueable: Yes\n\n"
                                            f"Critique: {feedback['critique']}"
                            }
                        )
    return data_list, rm_data_list


if __name__ == '__main__':
    train_sft_data, train_rm_data = extract_helpful_and_rm_data(r"train.jsonl(2)\train.jsonl")
    test_sft_data, test_rm_data = extract_helpful_and_rm_data(r"test.jsonl(2)\test.jsonl")
    DatasetDict({"train": Dataset.from_list(train_rm_data), "test":  Dataset.from_list(test_rm_data)}).push_to_hub("self-critiquing-helpful-rate")
    DatasetDict({"train": Dataset.from_list(train_sft_data), "test":  Dataset.from_list(test_sft_data)}).push_to_hub("self-critiquing-helpful-sft")

