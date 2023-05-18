# starting from https://github.com/openai/openai-cookbook/blob/main/examples/Get_embeddings.ipynb


import openai
from datasets import load_dataset, Dataset
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tqdm


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]



def get_unique_test_data(dataset):
    out_data = dict()
    for item in tqdm.tqdm(dataset['test']):
        # only use topic summarization tasks
        if not item['is_topic_based_summarization']:
            continue
        if item['id'] not in out_data.keys():
            out_data[item['id']] = {
                "id": item['id'],
                "split": item['split'],
                "time": item['time'],
                "labeler": item['labeler'],
                "is_topic_based_summarization": item['is_topic_based_summarization'],
                'prompt': item['prompt'],
                'responses': [item['response']]
            }
        else:
            out_data[item['id']]['responses'].append(item['response'])
    return list(out_data.values())



if __name__ == '__main__':
    dataset = load_dataset("dmayhem93/self-critiquing-base")
    to_be_embedded = get_unique_test_data(dataset)
    print(to_be_embedded[0])
    for i in tqdm.tqdm(range(len(to_be_embedded))):
        to_be_embedded[i]['embedding'] = get_embedding(to_be_embedded[i]['prompt'], model="text-embedding-ada-002")
    Dataset.from_list(to_be_embedded).push_to_hub("self-critiquing-base-topic-embeddings")
# print(len(embedding))