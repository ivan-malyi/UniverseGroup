from openai import OpenAI
import json
import pandas as pd
import os
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PROMPT_NAME = os.path.join(BASE_DIR, "prompt.txt")

class ClusterNamer:
    def __init__(self, table_path, save_dir):
        self.client = OpenAI()
        self.clusters = []

        try:
            self.df = pd.read_csv(table_path)
        except:
            raise TypeError("Enter correct path to csv table")

        self.save_dir = save_dir




    def _name_cluster(self, app_names: list) -> dict:
        with open(FILE_PROMPT_NAME, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        apps_text = "\n".join(app_names)
        # Substituting the list of applications
        prompt = prompt_template.replace("{apps_text}", apps_text)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}])

        content = response.choices[0].message.content
        content = content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(content)



    def _json_save(self, file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(self.clusters, f, ensure_ascii=False, indent=2)


    def _csv_save(self, file_name):
        rows = []
        for cluster in self.clusters:
            for app in cluster["apps"]:
                rows.append({
                    "AppName": app,
                    "SubNiche": cluster["niche_name"],
                    "Description": cluster["niche_description"]
                })

        pd.DataFrame(rows).to_csv(file_name, index=False)

    def process_clusters(self):
        unique_clusters = [c for c in self.df["cluster"].unique() if c != -1]

        for cluster_id in tqdm(unique_clusters, desc="Naming clusters"):
            app_names = self.df[self.df["cluster"] == cluster_id]["trackName"].tolist()
            niche = self._name_cluster(app_names)

            self.clusters.append({
                "niche_name": niche["name"],
                "niche_description": niche["description"],
                "apps": app_names
            })

        print(f"{len(self.clusters)} clusters processed")


    def __call__(self, json_name, csv_name):
        self.process_clusters()

        if len(self.clusters) == 0:
            raise TypeError("Empty clusters list - nothing to save")

        json_path = os.path.join(self.save_dir, json_name)
        csv_path= os.path.join(self.save_dir, csv_name)

        self._json_save(json_path)
        print(f"JSON saved to: {json_path}")
        self._csv_save(csv_path)
        print(f"CSV saved to: {csv_path}")

