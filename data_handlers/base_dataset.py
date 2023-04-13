from datasets import Dataset, DatasetDict
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
                if obj['data']['passage']['title'] is not None:
                    prompt = f"{obj['data']['passage']['title']}\n{obj['data']['passage']['text']}"
                else:
                    prompt = f"{obj['data']['passage']['text']}"
                data_list.append(
                    {
                        "id": obj['id'],
                        "split": obj['split'],
                        "time": obj['time'],
                        "labeler": obj['labeler'],
                        "is_topic_based_summarization": obj["is_topic_based_summarization"],
                        "prompt": prompt,
                        "response": f"Question: {obj['data']['questions'][i]['question']}\n\n"
                                    f"Answer: {obj['data']['questions'][i]['answer']}"
                    }
                )
    return data_list


if __name__ == '__main__':
    # From https://arxiv.org/abs/2206.05802 base task
    # train.jsonl from https://openaipublic.blob.core.windows.net/critiques/dataset/base/train.jsonl.gz
    # test.jsonl from https://openaipublic.blob.core.windows.net/critiques/dataset/base/test.jsonl.gz
    train_data = extract_data(r"train.jsonl\train.jsonl")
    test_data = extract_data(r"test.jsonl\test.jsonl")
    DatasetDict({"train": Dataset.from_list(train_data), "test":  Dataset.from_list(test_data)}).push_to_hub("self-critiquing-base")
