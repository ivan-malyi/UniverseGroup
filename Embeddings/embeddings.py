from langchain_openai import OpenAIEmbeddings
import numpy as np
import json
import pandas as pd
import os



class OpenAiEmbeddings:
    def __init__(self, save_dir: str):
        self.json_data = None
        self.df = None
        self.save_dir = save_dir


    def __call__(self, input_file: str):

        embeddings_file_path = os.path.join(self.save_dir, "embeddings.npy")
        df_file_path = os.path.join(self.save_dir, "DataFrame.csv")

        with open(input_file, "r", encoding="utf-8") as f:
            self.json_data = json.load(f)

        self.df = pd.DataFrame(self.json_data)

        # Merge text
        self.df["text"] = (self.df["trackName"] + " " + self.df["overview"] + " " +
                           self.df["features"].apply(lambda x: " ".join(x)))

        # Embeddings
        print("Vectorization started...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print(f"Vectorization finished!")

        # Save
        print("Saving in np file..  (this may take 1-2 minutes)")
        vectors = embeddings.embed_documents(self.df["text"].tolist())
        np.save(embeddings_file_path, np.array(vectors))
        self.df.to_csv(df_file_path, index=False)

        print(f"Saved in directory {self.save_dir}: \n\t 1.Embeddings: {embeddings_file_path} "
              f"\n\t 2.Table: {df_file_path}")
