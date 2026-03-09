from Embeddings import OpenAiEmbeddings
from Model import ClusterNamer
from Visualization import BrowserVisualization
import os
import numpy as np
import umap
import hdbscan
import pandas as pd
from dotenv import load_dotenv

SOURCE_PATH = "./Data/subscription_apps.json"
SAVE_DIR = "./Output"
TABLE_PATH = "./Output/DataFrame.csv"
EMBEDDINGS_PATH = "./Output/embeddings.npy"


def main():
    load_dotenv()  # load env variables (API Key for OpenAI)



    if not os.path.exists(EMBEDDINGS_PATH):
        print("Start OpenAI embeddings process")
        embedder = OpenAiEmbeddings(save_dir=SAVE_DIR)
        embedder(input_file=SOURCE_PATH)


    if not os.path.exists(os.path.join(SAVE_DIR, "results.csv")):
        # Vectors loading
        vectors = np.load(EMBEDDINGS_PATH)
        df = pd.read_csv(TABLE_PATH)

        # Reducing dimensionality UMAP (1536 -> 50)
        print("UMAP reducing dimensionality... (this may take 1-2 minutes)")
        reducer = umap.UMAP(n_components=10, random_state=42)
        reduced = reducer.fit_transform(vectors)
        print("UMAP done!")


        # Clustering by HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
        labels = clusterer.fit_predict(reduced)

        # Add cluster labels in DataFrame
        df["cluster"] = labels

        print(f"Cluster count: {len(set(labels)) - (1 if -1 in labels else 0)}")
        print(f"Noise (not in cluster): {sum(labels == -1)}")

        df.to_csv(TABLE_PATH, index=False)


        cluster_obj = ClusterNamer(table_path=TABLE_PATH, save_dir=SAVE_DIR)
        cluster_obj("results.json", "results.csv")

        results_df = pd.read_csv(os.path.join(SAVE_DIR, "results.csv"))
        df = df.merge(results_df[["AppName", "SubNiche"]], left_on="trackName", right_on="AppName", how="left")
        df.to_csv(TABLE_PATH, index=False)

    # Optional block with Plotly visualization
    visualization_obj = BrowserVisualization(source_dir=SAVE_DIR)
    visualization_obj(table_name="DataFrame.csv", embeddings_name="embeddings.npy")




if __name__ == '__main__':
    main()