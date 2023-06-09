from datasets import Dataset, load_dataset
import numpy as np
from sklearn.cluster import KMeans


if __name__ == "__main__":
    dataset = load_dataset("dmayhem93/self-critiquing-base-selected-900")
    out_list = list()
    for item in dataset['train']:
        out_list.append({
            "id": item["id"],
            "split": item["split"],
            "time": item["time"],
            "labeler": item["labeler"],
            "is_topic_based_summarization": item["is_topic_based_summarization"],
            "prompt": item["prompt"] + item['responses'][0].split("Answer:")[0] + "Answer:",
        })
    Dataset.from_list(out_list).push_to_hub("self-critiquing-base-selected-900-prompt-continuation")