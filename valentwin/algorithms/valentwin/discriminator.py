import pandas as pd
import random
import torch

from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm
from typing import List, Union

from valentwin.algorithms.valentwin.utils import cos_sim
from valentwin.embedder.text_embedder import HFTextEmbedder


class ABXDiscriminator:
    """
    ABX Discriminator used for evaluation during contrastive training
    """

    def __init__(self, text_embedder: HFTextEmbedder):
        self.text_embedder = text_embedder

    def discriminate_with_contrastive_training_file(self, input_file_path: str, output_file_path: str = None,
                                                    a_column: str = "sent0", b_column: str = "hard_neg",
                                                    x_column: str = "sent1", prediction_column: str = "prediction"):
        df = pd.read_csv(input_file_path)

        # batch per 512 to avoid memory issues --> DEPRECATED
        # now we use the text_embedder's batch size
        predictions = []
        for i in tqdm(range(0, len(df), self.text_embedder.inference_batch_size)):
            a_embeddings = self.text_embedder.get_sentence_embeddings(df[a_column][i:i+self.text_embedder.inference_batch_size].astype(str).tolist())
            b_embeddings = self.text_embedder.get_sentence_embeddings(df[b_column][i:i+self.text_embedder.inference_batch_size].astype(str).tolist())
            x_embeddings = self.text_embedder.get_sentence_embeddings(df[x_column][i:i+self.text_embedder.inference_batch_size].astype(str).tolist())

            a_x_sim = torch.diagonal(cos_sim(a_embeddings, x_embeddings))
            b_x_sim = torch.diagonal(cos_sim(b_embeddings, x_embeddings))

            predictions.extend(["a" if a_x_sim > b_x_sim else "b" if b_x_sim > a_x_sim else random.choice(["a", "b"])
                                for a_x_sim, b_x_sim in zip(a_x_sim, b_x_sim)])

        df[prediction_column] = predictions
        if output_file_path:
            df.to_csv(output_file_path, index=False)
        return df

    def evaluate_discriminations(self, input_file_path: str, output_file_path: str, prediction_columns: List[str],
                                correct_labels: Union[str, List[str]] = "a", return_accuracy_only=True):
        df = pd.read_csv(input_file_path)
        if isinstance(correct_labels, str):
            correct_labels = [correct_labels] * len(df)

        eval_results = {}
        for prediction_column in prediction_columns:
            print(f"Classification report for {prediction_column}")
            print(classification_report(correct_labels, df[prediction_column]))
            print("\n")
            report = classification_report(correct_labels, df[prediction_column], output_dict=True)
            if return_accuracy_only:
                eval_results[prediction_column] = report["accuracy"]
            else:
                eval_results[prediction_column] = report

        eval_results_df = pd.DataFrame(eval_results)
        eval_results_df.to_csv(output_file_path, index=False)
        return eval_results

    def evaluate_discrimination(self, input_file_path: str = None, prediction_df: pd.DataFrame = None,
                                prediction_column: str = "prediction",
                                correct_labels: Union[str, List[str]] = "a", return_accuracy_only=True):
        if input_file_path is not None:
            df = pd.read_csv(input_file_path)
        else:
            df = prediction_df
        if isinstance(correct_labels, str):
            correct_labels = [correct_labels] * len(df)

        eval_results = {}
        # print(f"Classification report for {prediction_column}")
        # print(classification_report(correct_labels, df[prediction_column]))
        # print("\n")
        if return_accuracy_only:
            eval_results["accuracy"] = accuracy_score(correct_labels, df[prediction_column])
        else:
            eval_results = classification_report(correct_labels, df[prediction_column], output_dict=True)

        return eval_results


