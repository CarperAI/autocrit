import datasets
import json


if __name__ == '__main__':
    with open("responses.json", encoding='utf-8') as f:
        responses = json.load(f)
    train_list = list()
    for item in responses.values():
        good_count = 0
        for good in item['good']:
            if "openai" in good.lower():
                continue
            else:
                train_list.append({'prompt': item['prompt'], 'response': good, 'type': 'good'})
            if 'stablebeluga' in good.lower():
                good_count += 1
            elif 'carper' in good.lower():
                good_count += 1
        if len(item['good']) == good_count:
            for bad in item['bad']:
                train_list.append({'prompt': item['prompt'], 'response': bad, 'type': 'bad'})
    print(train_list[:10])
    datasets.DatasetDict(
        {"train": datasets.Dataset.from_list(train_list)}
    ).push_to_hub("stablebeluga-knowledge-name-creator", private=True)