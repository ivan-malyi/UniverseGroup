import umap
import plotly.express as px
import numpy as np
import pandas as pd
import os



class BrowserVisualization:
    def __init__(self, source_dir: str):
        self.source_dir = source_dir


    def __call__(self, table_name: str, embeddings_name: str):
        table_path = os.path.join(self.source_dir, table_name)
        embeddings = os.path.join(self.source_dir, embeddings_name)
        df = pd.read_csv(table_path)
        vectors = np.load(embeddings)


        # Compress to 2D for visualization
        reducer_2d = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer_2d.fit_transform(vectors)

        # Add coordinates to the DataFrame
        df["x"] = embedding_2d[:, 0]
        df["y"] = embedding_2d[:, 1]
        df["cluster_label"] = df["cluster"].astype(str)  # for color

        # Building a schedule
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster_label",
            hover_data=["trackName", "SubNiche"],  # Shows the name when hovered over
            title="Mobile Apps Clustering",
            width=1200,
            height=800
        )
        fig.update_layout(showlegend=False) # It looks better without a legend.
        fig.show()  # opens in browser