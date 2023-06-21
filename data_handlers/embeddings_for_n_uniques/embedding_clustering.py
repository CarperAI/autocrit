from datasets import Dataset, load_dataset
import numpy as np
from sklearn.cluster import KMeans


if __name__ == "__main__":
    dataset = load_dataset("dmayhem93/self-critiquing-base-topic-embeddings")
    np_data = list()
    for i in range(len(dataset["train"])):
        np_data.append(np.array(dataset["train"][i]["embedding"]))
    np_data = np.vstack(np_data)
    kmeans = KMeans(n_clusters=900, init="k-means++", random_state=42)
    kmeans.fit(np_data)
    out_list = []
    labels = kmeans.labels_
    out_list = [None for i in range(900)]
    for i in range(len(labels)):
        if out_list[int(labels[i])] is None:
            out_list[int(labels[i])] = dataset["train"][i]
    Dataset.from_list(out_list).push_to_hub("self-critiquing-base-selected-900")