import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.metrics import silhouette_score
from typing import List
from umap import UMAP

from valentwin.embedder.text_embedder import TextEmbedder


class EmbeddingVisualizer():
    def __init__(self, text_col, label_col, other_cols=None):
        self.text_col = text_col
        self.label_col = label_col
        self.other_cols = other_cols if other_cols else []

    def _visualize_2d(self, embeddings_2d: np.array, df: pd.DataFrame, hover_cols: List[str],
                      size_col: str, title: str, color_col: str = 'label', ):
        """
        Visualize embeddings in 2D with plotly
        :param df: pd.DataFrame containing meta-data (data relating to hover, color, size, etc.)
                    The order of the DataFrame must match the order of the embeddings.
        :param embeddings_2d: 2D embeddings, shape (n, 2)
        :param hover_cols: Columns to display in the hover tooltip
        :param color_col: Column to use for coloring the points
        :param size_col: Column to use for sizing the points
        :param title: Title of the plot
        :return: px.scatter plot
        """
        assert len(df) == embeddings_2d.shape[0]
        assert embeddings_2d.shape[1] == 2

        vis_cols = []
        if color_col:
            vis_cols.append(color_col)
        if size_col:
            vis_cols.append(size_col)
        if hover_cols:
            vis_cols.extend(hover_cols)

        df_vis = df[vis_cols].copy()
        df_vis['x'] = embeddings_2d[:, 0]
        df_vis['y'] = embeddings_2d[:, 1]
        fig = px.scatter(df_vis, x='x', y='y',
                         color=color_col, size=size_col,
                         hover_data=hover_cols, title=title)
        return fig

    def _visualize_3d(self, embeddings_3d: np.array, df: pd.DataFrame, hover_cols: List[str],
                      size_col: str, title: str, color_col: str = 'label', ):
        """
        Visualize embeddings in 3D with plotly
        :param df: pd.DataFrame containing meta-data (data relating to hover, color, size, etc.)
                    The order of the DataFrame must match the order of the embeddings.
        :param embeddings_3d: 3D embeddings, shape (n, 3)
        :param hover_cols: Columns to display in the hover tooltip
        :param color_col: Column to use for coloring the points
        :param size_col: Column to use for sizing the points
        :param title: Title of the plot
        :return: px.scatter plot
        """
        assert len(df) == embeddings_3d.shape[0]
        assert embeddings_3d.shape[1] == 3

        vis_cols = []
        if color_col:
            vis_cols.append(color_col)
        if size_col:
            vis_cols.append(size_col)
        if hover_cols:
            vis_cols.extend(hover_cols)

        df_vis = df[vis_cols].copy()
        df_vis['x'] = embeddings_3d[:, 0]
        df_vis['y'] = embeddings_3d[:, 1]
        df_vis['z'] = embeddings_3d[:, 2]
        fig = px.scatter_3d(df_vis, x='x', y='y', z='z',
                            color=color_col, size=size_col,
                            hover_data=hover_cols, title=title)
        return fig

    def visualize_data(self, embedder: TextEmbedder, dataset: pd.DataFrame, dim=2):
        umap = UMAP(
            random_state=1,
            metric="cosine",
            n_components=dim,
        )

        embeddings = embedder.get_sentence_embeddings(dataset[self.text_col].tolist()).cpu()
        embeddings_umap = umap.fit_transform(embeddings)

        if dim == 2:
            return self._visualize_2d(embeddings_umap, dataset, hover_cols=[self.text_col]+self.other_cols,
                                      color_col=self.label_col, size_col=None, title='')
        elif dim == 3:
            return self._visualize_3d(embeddings_umap, dataset, hover_cols=[self.text_col]+self.other_cols,
                                      color_col=self.label_col, size_col=None, title='')
        else:
            raise ValueError("dim must be 2 or 3")

    def get_silhouette_score(self, embedder: TextEmbedder, dataset: pd.DataFrame, metric: str = "cosine"):
        embeddings = embedder.get_sentence_embeddings(dataset[self.text_col].tolist()).cpu()
        return silhouette_score(embeddings, dataset[self.label_col].tolist(), metric=metric)
