from datasets import Dataset
import jsonlines


def extract_data(file_path):
    data_list = list()
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            for i in range(len(obj['data']['questions'])):
                if obj['data']['questions'][i]['question'] == '':
                    continue
                if obj['data']['questions'][i]['answer'] == '':
                    continue
                data_list.append(
                    {
                        "id": obj['id'],
                        "split": obj['split'],
                        "time": obj['time'],
                        "labeler": obj['labeler'],
                        "is_topic_based_summarization": obj["is_topic_based_summarization"],
                        "prompt": f"{obj['data']['passage']['title']}\n"
                                  f"{obj['data']['passage']['text']}",
                        "question": f"Question: {obj['data']['questions'][i]['question']}\n"
                                    f"Answer: {obj['data']['questions'][i]['answer']}"
                    }
                )
    return data_list


if __name__ == '__main__':
    train_data = extract_data(r"train.jsonl\train.jsonl")
    test_data = extract_data(r"test.jsonl\test.jsonl")
    Dataset.from_list(train_data).push_to_hub("self-critiquing-base-train")
    Dataset.from_list(test_data).push_to_hub("self-critiquing-base-test")
