import os
import sys
import glob
import pickle
import gzip
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def load_experiment_results(results_dir):
    """Loads all results_dict_*.pkl or .pkl.gz files from the results directory."""
    pkl_files = glob.glob(os.path.join(results_dir, "results_dict_*.pkl"))
    gz_files = glob.glob(os.path.join(results_dir, "results_dict_*.pkl.gz"))
    all_files = pkl_files + gz_files
    
    if not all_files:
        raise ValueError(f"No results_dict_*.pkl or .pkl.gz files found in {results_dir}")
        
    print(f"Found {len(all_files)} result files. Loading...")
    
    records = []
    for fpath in tqdm(all_files, desc="Loading pickles"):
        try:
            if fpath.endswith(".gz"):
                with gzip.open(fpath, "rb") as f:
                    res = pickle.load(f)
            else:
                with open(fpath, "rb") as f:
                    res = pickle.load(f)
            
            # Extract key information safely
            question = res.get("question", "")
            question_id = res.get("question_id", None)
            best_answer = res.get("best_answer", "")
            
            # Extract correctness dict and BLEURT score
            corr_dict = res.get("correctness_dict", {})
            bleurt_score = corr_dict.get("bleurt", None)
            
            # Extract greedy/most_likely answer text
            greedy_answer = ""
            if "sdlg" in res and "generations" in res["sdlg"]:
                gens = res["sdlg"]["generations"]
                if gens and "generation_text" in gens[0]:
                    greedy_answer = gens[0]["generation_text"][0]
            elif "generations" in res:
                # Fallback for other formats
                gens = res["generations"]
                if gens and "generation_text" in gens[0]:
                    greedy_answer = gens[0]["generation_text"][0]
                    
            records.append({
                "question_id": question_id,
                "question": question,
                "best_answer": best_answer,
                "greedy_answer": greedy_answer,
                "bleurt": bleurt_score,
                "file_path": fpath
            })
        except Exception as e:
            print(f"Warning: Failed to load {fpath}: {e}")
            
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} samples successfully.")
    return df


def fetch_dataset_metadata(df, dataset_name):
    """Loads dataset from Hugging Face to map questions to pre-defined categories."""
    if not HAS_DATASETS:
        print("HuggingFace 'datasets' package is not installed. Skipping metadata mapping.")
        return df
        
    print(f"Fetching '{dataset_name}' dataset from Hugging Face for topic metadata...")
    try:
        if "truthful" in dataset_name.lower():
            ds = load_dataset("truthfulqa/truthful_qa", "generation")
            val_set = ds["validation"]
            # Create a lookup mapping from question text to category
            lookup = {sample["question"].strip().lower(): sample.get("category", "Unknown") for sample in val_set}
            df["dataset_category"] = df["question"].apply(lambda q: lookup.get(q.strip().lower(), "Unknown"))
            
        elif "science" in dataset_name.lower():
            ds = load_dataset("tasksource/ScienceQA_text_only")
            test_set = ds["test"]
            
            # ScienceQA has rich metadata like topic, category, subject
            lookup_topic = {}
            lookup_cat = {}
            lookup_sub = {}
            for sample in test_set:
                q = sample["question"].strip().lower()
                lookup_topic[q] = sample.get("topic", "Unknown")
                lookup_cat[q] = sample.get("category", "Unknown")
                lookup_sub[q] = sample.get("subject", "Unknown")
                
            df["dataset_topic"] = df["question"].apply(lambda q: lookup_topic.get(q.strip().lower(), "Unknown"))
            df["dataset_category"] = df["question"].apply(lambda q: lookup_cat.get(q.strip().lower(), "Unknown"))
            df["dataset_subject"] = df["question"].apply(lambda q: lookup_sub.get(q.strip().lower(), "Unknown"))
        else:
            print(f"Dataset '{dataset_name}' not specifically recognized for pre-defined metadata mapping.")
    except Exception as e:
        print(f"Error fetching dataset metadata: {e}")
        
    return df


def generate_question_embeddings(df, model_name):
    """Generates embeddings for questions using SentenceTransformers."""
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers package is required. Install it using 'pip install sentence-transformers'.")
        
    print(f"Loading sentence embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    questions = df["question"].tolist()
    print("Generating embeddings...")
    embeddings = model.encode(questions, show_progress_bar=True)
    return embeddings


def perform_clustering(df, embeddings, n_clusters=10):
    """Clusters question embeddings and generates descriptive labels for each cluster."""
    print(f"Clustering questions into {n_clusters} clusters using KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    df["cluster"] = cluster_ids
    
    # Calculate distances of questions to cluster centers to find representative ones
    cluster_centers = kmeans.cluster_centers_
    distances = []
    for i, emb in enumerate(embeddings):
        c_id = cluster_ids[i]
        center = cluster_centers[c_id]
        dist = np.linalg.norm(emb - center)
        distances.append(dist)
    df["dist_to_center"] = distances
    
    # Generate automatic topic names for clusters
    cluster_topics = {}
    print("\nDetermining cluster topics...")
    
    for c_id in range(n_clusters):
        c_df = df[df["cluster"] == c_id]
        
        # 1. Representative questions (closest to centroid)
        rep_questions = c_df.nsmallest(3, "dist_to_center")["question"].tolist()
        
        # 2. Extract top terms using TF-IDF inside this cluster vs others
        tfidf = TfidfVectorizer(stop_words='english', max_features=10)
        # Combine all questions
        all_questions = df["question"].tolist()
        # Mark questions in current cluster vs others
        in_cluster_text = " ".join(c_df["question"])
        
        # Simple TF-IDF term selection
        try:
            tfidf.fit(all_questions)
            feature_names = tfidf.get_feature_names_out()
            cluster_tfidf = tfidf.transform([in_cluster_text]).toarray()[0]
            top_indices = np.argsort(cluster_tfidf)[::-1][:4]
            top_terms = [feature_names[idx] for idx in top_indices if cluster_tfidf[idx] > 0]
        except Exception:
            top_terms = []
            
        # 3. If dataset metadata is available, find the most frequent categories
        meta_summary = ""
        if "dataset_category" in df.columns:
            top_cat = c_df["dataset_category"].value_counts().index
            if len(top_cat) > 0 and top_cat[0] != "Unknown":
                meta_summary = top_cat[0]
            elif "dataset_topic" in df.columns:
                top_topic = c_df["dataset_topic"].value_counts().index
                if len(top_topic) > 0 and top_topic[0] != "Unknown":
                    meta_summary = top_topic[0]
                    
        # Define Topic Label
        if meta_summary:
            topic_label = f"{meta_summary} (Cluster {c_id})"
        elif top_terms:
            topic_label = f"Keywords: {', '.join(top_terms[:2])} (Cluster {c_id})"
        else:
            topic_label = f"Topic Cluster {c_id}"
            
        cluster_topics[c_id] = topic_label
        
        print(f"\nCluster {c_id} -> Label: '{topic_label}'")
        print("  Top representative questions:")
        for rq in rep_questions:
            print(f"    - {rq}")
            
    df["cluster_topic"] = df["cluster"].map(cluster_topics)
    return df, cluster_topics, kmeans


def run_dimensionality_reduction(embeddings, method="tsne"):
    """Reduces embeddings to 2D for visualization."""
    print(f"Reducing dimensions to 2D using {method.upper()}...")
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        coords = reducer.fit_transform(embeddings)
    elif method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    return coords


def create_visualizations(df, coords, bleurt_threshold, output_dir):
    """Creates static (matplotlib) and interactive (plotly) plots."""
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    
    # Determine correctness based on threshold
    df["correct"] = df["bleurt"] >= bleurt_threshold
    df["Correctness"] = df["correct"].map({True: "Correct (Score >= Thresh)", False: "Incorrect (Score < Thresh)"})
    
    # Fill in any NaN BLEURT values defensively
    df["bleurt_display"] = df["bleurt"].apply(lambda s: f"{s:.3f}" if pd.notnull(s) else "N/A")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Static plot colored by cluster
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df, x="x", y="y", hue="cluster_topic", 
        palette="tab20", alpha=0.8, edgecolor="none", s=35
    )
    plt.title("Question Semantic Clusters (Topics)", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Topics")
    plt.tight_layout()
    static_cluster_path = os.path.join(output_dir, "question_clusters_topics.png")
    plt.savefig(static_cluster_path, dpi=150)
    plt.close()
    print(f"Saved static cluster plot to: {static_cluster_path}")
    
    # Plot 2: Static plot colored by correctness
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df, x="x", y="y", hue="Correctness",
        palette={"Correct (Score >= Thresh)": "#4CAF50", "Incorrect (Score < Thresh)": "#F44336"},
        alpha=0.8, edgecolor="none", s=35
    )
    plt.title(f"LLM Correctness Map (BLEURT Threshold: {bleurt_threshold})", fontsize=14, fontweight="bold")
    plt.legend(loc='upper right')
    plt.tight_layout()
    static_corr_path = os.path.join(output_dir, "question_clusters_correctness.png")
    plt.savefig(static_corr_path, dpi=150)
    plt.close()
    print(f"Saved static correctness plot to: {static_corr_path}")
    
    # Plot 3: Interactive Plotly plot (if available)
    if HAS_PLOTLY:
        # Hover info columns
        hover_data = ["question", "best_answer", "greedy_answer", "bleurt_display", "cluster_topic"]
        if "dataset_category" in df.columns:
            hover_data.append("dataset_category")
            
        fig = px.scatter(
            df, x="x", y="y", 
            color="Correctness",
            color_discrete_map={"Correct (Score >= Thresh)": "#2E7D32", "Incorrect (Score < Thresh)": "#C62828"},
            hover_name="question_id",
            hover_data={col: True for col in hover_data},
            title=f"Interactive LLM Correctness Map (BLEURT Threshold: {bleurt_threshold})"
        )
        # Update layout
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            legend_title_text="Evaluation Status"
        )
        interactive_path = os.path.join(output_dir, "interactive_correctness_dashboard.html")
        fig.write_html(interactive_path)
        print(f"Saved interactive HTML dashboard to: {interactive_path}")


def analyze_model_biases(df, bleurt_threshold, output_dir):
    """Computes per-topic bias metrics and saves a summary report."""
    print("\n" + "=" * 50)
    print("BIAS SUMMARY REPORT BY TOPIC CLUSTER")
    print("=" * 50)
    
    # Group by cluster to calculate metrics
    summary_data = []
    
    grouped = df.groupby("cluster")
    for c_id, group in grouped:
        total_questions = len(group)
        correct_count = group["correct"].sum()
        accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0.0
        avg_bleurt = group["bleurt"].mean()
        
        # Determine dominant dataset category
        if "dataset_category" in group.columns:
            top_cats = group["dataset_category"].value_counts()
            dominant_cat = f"{top_cats.index[0]} ({top_cats.values[0]}/{total_questions})" if len(top_cats) > 0 else "N/A"
        else:
            dominant_cat = "N/A"
            
        topic_label = group["cluster_topic"].iloc[0]
        
        summary_data.append({
            "cluster_id": c_id,
            "topic_label": topic_label,
            "dominant_dataset_category": dominant_cat,
            "total_questions": total_questions,
            "correct_predictions": correct_count,
            "incorrect_predictions": total_questions - correct_count,
            "accuracy_pct": accuracy,
            "avg_bleurt": avg_bleurt
        })
        
        print(f"Topic: {topic_label}")
        print(f"  Accuracy: {accuracy:.1f}% ({correct_count}/{total_questions}) | Avg BLEURT: {avg_bleurt:.3f}")
        print(f"  Dominant Dataset Category: {dominant_cat}")
        print("-" * 50)
        
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="accuracy_pct")  # Sort by worst performing first to highlight biases
    
    report_csv_path = os.path.join(output_dir, "topic_bias_report.csv")
    summary_df.to_csv(report_csv_path, index=False)
    print(f"Saved tabular bias report to: {report_csv_path}")
    
    # Save a markdown report too
    report_md_path = os.path.join(output_dir, "topic_bias_report.md")
    with open(report_md_path, "w") as f:
        f.write("# LLM Topic Bias Analysis Report\n\n")
        f.write(f"- **BLEURT Threshold for correctness**: `{bleurt_threshold}`\n")
        f.write(f"- **Total evaluated questions**: `{len(df)}`\n")
        f.write(f"- **Overall Model Accuracy**: `{df['correct'].mean() * 100:.2f}%`\n\n")
        f.write("## Performance by Topic Cluster (Sorted from Worst to Best)\n")
        f.write("This table shows the model's accuracy on different semantic groups of questions. Clusters with lower accuracy highlight the model's biases/weaknesses.\n\n")
        
        f.write("| Cluster ID | Topic Label | Dominant Category | Total Questions | Accuracy (%) | Avg BLEURT |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in summary_df.iterrows():
            f.write(f"| {row['cluster_id']} | {row['topic_label']} | {row['dominant_dataset_category']} | {row['total_questions']} | {row['accuracy_pct']:.1f}% | {row['avg_bleurt']:.3f} |\n")
            
        f.write("\n\n## Representative Questions per Cluster (Top 3 closest to centroid)\n")
        for c_id in sorted(df["cluster"].unique()):
            c_df = df[df["cluster"] == c_id]
            topic_label = c_df["cluster_topic"].iloc[0]
            rep_questions = c_df.nsmallest(3, "dist_to_center")["question"].tolist()
            f.write(f"### {topic_label}\n")
            for rq in rep_questions:
                f.write(f"- {rq}\n")
            f.write("\n")
            
    print(f"Saved Markdown bias report to: {report_md_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize LLM performance biases using question clustering.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the directory containing results_dict_*.pkl files.")
    parser.add_argument("--dataset", type=str, required=True, choices=["truthfulqa", "scienceqa"], help="Name of the dataset evaluated.")
    parser.add_argument("--bleurt_threshold", type=float, default=-0.3, help="BLEURT threshold above which a response is marked correct.")
    parser.add_argument("--num_clusters", type=int, default=12, help="Number of semantic topics (clusters) to discover.")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="Hugging Face sentence-transformer model for embedding questions.")
    parser.add_argument("--reduction_method", type=str, default="tsne", choices=["tsne", "pca"], help="Dimensionality reduction method.")
    parser.add_argument("--output_dir", type=str, default="./bias_analysis", help="Directory where visual and tabular reports will be saved.")
    
    args = parser.parse_args()
    
    # 1. Load results
    try:
        df = load_experiment_results(args.results_dir)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
        
    # Remove rows with no BLEURT scores (eval failed or wasn't run)
    initial_len = len(df)
    df = df.dropna(subset=["bleurt"])
    if len(df) < initial_len:
        print(f"Filtered out {initial_len - len(df)} samples that were missing BLEURT score.")
    
    if len(df) == 0:
        print("Error: No samples with valid BLEURT scores were found. Cannot analyze.")
        return
        
    # 2. Fetch ground-truth dataset categories
    df = fetch_dataset_metadata(df, args.dataset)
    
    # 3. Create question embeddings
    try:
        embeddings = generate_question_embeddings(df, args.embedding_model)
    except Exception as e:
        print(f"Error generating question embeddings: {e}")
        return
        
    # 4. Cluster questions and assign topic labels
    df, cluster_topics, kmeans = perform_clustering(df, embeddings, n_clusters=args.num_clusters)
    
    # 5. Dimension reduction
    coords = run_dimensionality_reduction(embeddings, method=args.reduction_method)
    
    # 6. Visualization
    create_visualizations(df, coords, args.bleurt_threshold, args.output_dir)
    
    # 7. Bias analysis and reports
    analyze_model_biases(df, args.bleurt_threshold, args.output_dir)


if __name__ == "__main__":
    main()
