# 📱 Mobile App Sub-Niche Clustering

Automatic segmentation of 4200+ mobile applications into granular sub-niches using OpenAI Embeddings + UMAP + HDBSCAN.

## 🎯 Goal

Replace manual competitor research with an algorithm that automatically groups mobile apps into sub-niches where every app is a **direct competitor** of others in the same group.

## ⚙️ Pipeline

```
JSON Dataset → OpenAI Embeddings → UMAP (dim reduction) → HDBSCAN (clustering) → GPT-4o-mini (naming) → JSON / CSV
```

## 🗂️ Project Structure

```
├── Data/                   # Raw dataset (4200+ apps)
├── Embeddings/             # Vectorization via OpenAI text-embedding-3-small
├── Model/                  # Cluster naming via GPT-4o-mini
├── Visualization/          # Interactive Plotly scatter plot
├── Output/                 # Results: embeddings.npy, results.json, results.csv
└── main.py                 # Entry point
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 3. Run
python main.py
```

## 📊 Output

| File | Description |
|---|---|
| `results.json` | Niches with name, description and list of apps |
| `results.csv` | AppName \| SubNiche \| Description |

**Example:**
```json
{
  "niche_name": "AI Tattoo Design",
  "niche_description": "Apps that generate tattoo designs using AI.",
  "apps": ["Tattoo AI", "InkDNA", "AI Tattoo Generator"]
}
```

## 🧠 Tech Stack

| Tool | Purpose |
|---|---|
| `text-embedding-3-small` | Semantic vectorization of app descriptions |
| `UMAP` | Dimensionality reduction (1536 → 10) |
| `HDBSCAN` | Density-based clustering |
| `GPT-4o-mini` | Niche naming and description |
| `Plotly` | Interactive cluster visualization |


<img width="1200" height="800" alt="newplot" src="https://github.com/user-attachments/assets/d1f76c86-e62d-4a52-8e77-fa61878fb79c" />

