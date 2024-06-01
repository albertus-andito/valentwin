import numpy as np
import os
import pandas as pd

from scipy.spatial.distance import pdist

from valentwin.embedder.text_embedder import HFTextEmbedder
from valentwin.embedder.visualizer import EmbeddingVisualizer

data_root_dir = "../../data"


def prepare_dataset(dataset_name, dataset_collection):
    dataset_files = os.listdir(os.path.join(data_root_dir, dataset_collection, dataset_name, "sample", "100-test"))

    gt_labels = {}
    gt_df = pd.read_csv(
        os.path.join(data_root_dir, dataset_collection, dataset_name, "ground-truth-mapping", "ground-truth.csv"),
        index_col=0)

    for i, row in gt_df.iterrows():
        col_1 = str(row["source_table"]) + "-" + str(row["source_column"])
        col_2 = str(row["target_table"]) + "-" + str(row["target_column"])

        found_key = None
        for key, value_list in gt_labels.items():
            if col_1 in value_list or col_2 in value_list:
                found_key = key
                break
        if found_key is None:
            found_key = col_1

        value_list = gt_labels.get(found_key, set())

        value_list.add(col_1)
        value_list.add(col_2)

        gt_labels[found_key] = value_list

    dataset_text = []
    dataset_labels = []
    dataset_fnames = []
    dataset_column_names = []
    for fname in dataset_files:
        df = pd.read_csv(os.path.join(data_root_dir, dataset_collection, dataset_name, "sample", "100-test", fname))
        for col in df.columns:
            unique_texts = df[col].astype(str).unique().tolist()
            found_key = None
            for key, value_list in gt_labels.items():
                if fname.replace(".csv", "") + "-" + col in value_list:
                    found_key = key
                    break
            if found_key is None:
                continue
            dataset_text.extend(unique_texts)
            dataset_labels.extend([found_key] * len(unique_texts))
            dataset_fnames.extend([fname.replace(".csv", "")] * len(unique_texts))
            dataset_column_names.extend([col] * len(unique_texts))

    dataset = pd.DataFrame({"text": dataset_text, "label": dataset_labels, "dataset_name": dataset_fnames,
                            "column_name": dataset_column_names})
    return dataset


def calculate_mean_similarity(model, dataset):
    labels = dataset["label"].unique()
    sims = []
    for label in dataset["label"].unique():
        embeddings = model.get_sentence_embeddings(dataset[dataset["label"] == label]["text"].tolist()).cpu()
        pairwise_cosine_distances = pdist(embeddings, metric='cosine')
        pairwise_cosine_similarities = 1 - pairwise_cosine_distances
        mean_similarity = np.mean(pairwise_cosine_similarities)
        sims.append(mean_similarity)

    dataset_mean_sim = np.mean(sims)
    sims_df = pd.DataFrame.from_dict({"label": labels, "mean_similarity": sims})
    return sims_df, dataset_mean_sim


if __name__ == "__main__":
    viz = EmbeddingVisualizer("text", "label", ["dataset_name", "column_name"])

    all_silhouette_scores = []
    all_mean_similarities = []
    all_scores_df = []

    model_keys = ["simcse",
                  "valentwin-noneg", "valentwin-neg",
                  "valentwin-neginter"]

    alite_datasets = ["25ksep11", "500spend", "1009ipopayments", "amo-ame", "cihr", "chicago_parks", "DCMS_NHM_NHM",
                      "organogram-junior", "school_report", "stockport_contracts"]
    magellan_datasets = ["academic_papers", "books", "cosmetics", "movies", "restaurants"]

    collections = [alite_datasets, magellan_datasets]
    collection_names = ["alite", "magellan"]

    for collection_name, collection in zip(collection_names, collections):
        for dataset_name in collection:
            dataset = prepare_dataset(dataset_name, collection_name)
            model_names = ["princeton-nlp/sup-simcse-roberta-base",
                           f"albertus-andito/valentwin-{dataset_name}-n-100-hn-10-selective-noneg-lr-3e5-bs-512",
                           f"albertus-andito/valentwin-{dataset_name}-n-100-hn-10-selective-neg-lr-3e5-bs-512",
                           f"albertus-andito/valentwin-{dataset_name}-n-100-hn-10-selective-neginter-lr-3e5-bs-512",
                           ]
            silhouette_scores = []
            mean_similarities = []
            for model_name in model_names:
                model = HFTextEmbedder(model_name, use_auth_token=True, use_cache=True, device="cuda:0")
                silhouette_scores.append(viz.get_silhouette_score(model, dataset, "cosine"))
                mean_similarities.append(calculate_mean_similarity(model, dataset)[1])
                model = None
                scores_df = pd.DataFrame.from_dict(
                    {"model": model_keys, "silhouette_score": silhouette_scores, "mean_similarity": mean_similarities,
                     "integration_set": dataset_name})
                all_scores_df.append(scores_df)

    concatenated_df = pd.concat(all_scores_df, axis=0, ignore_index=True)
    concatenated_df.to_csv("valentwin_silhouette_scores.csv")