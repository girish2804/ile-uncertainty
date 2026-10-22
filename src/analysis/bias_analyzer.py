import os
import sys

# Ensure all code directories (src, src/experiments, src/analysis, and root) are in sys.path
# This allows pickle to resolve imports like 'utils', 'args', and 'seq_lvl_imp'
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/analysis
src_dir = os.path.dirname(current_dir)                    # src/
experiments_dir = os.path.join(src_dir, "experiments")    # src/experiments
project_root = os.path.dirname(src_dir)                   # project root

for path in [project_root, src_dir, experiments_dir, current_dir]:
    if path not in sys.path:
        sys.path.append(path)

import glob
import pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Custom stop words specifically for question datasets to filter out structural noise
QUESTION_STOP_WORDS = {
    'what', 'how', 'why', 'who', 'where', 'when', 'which', 'whom',
    'does', 'do', 'did', 'is', 'are', 'was', 'were', 'can', 'could',
    'should', 'would', 'will', 'shall', 'has', 'have', 'had', 'people',
    'say', 'think', 'know', 'believe', 'make', 'happen', 'called',
    'true', 'false', 'correct', 'incorrect', 'mean', 'difference',
    'many', 'much', 'some', 'any', 'every', 'all', 'one', 'two',
    'years', 'time', 'world', 'earth', 'country', 'state', 'city', 'use'
}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(QUESTION_STOP_WORDS))

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from IPython.display import display, Markdown
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class BiasAnalyzer:
    """Class to analyze and visualize LLM biases on various topics in a Jupyter Notebook."""
    
    def __init__(self, base_results_dir="result", embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self._embedder = None
        
        # Smart detection of the results base directory
        candidates = []
        if base_results_dir:
            candidates.append(base_results_dir)
            candidates.append(os.path.abspath(base_results_dir))
            
        # Add relative path from this script's directory: src/analysis/ -> src/experiments/result
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "result")))
        # Add project level results directory candidate
        candidates.append("result")
        candidates.append("results")
        candidates.append("../experiments/result")
        
        resolved_dir = base_results_dir
        for cand in candidates:
            if cand and os.path.exists(cand):
                try:
                    entries = os.listdir(cand)
                    # If this directory has subdirectories with truthfulqa or scienceqa, resolve it!
                    if any(os.path.isdir(os.path.join(cand, e)) and ("truthful" in e.lower() or "science" in e.lower()) for e in entries):
                        resolved_dir = cand
                        break
                except Exception:
                    pass
                    
        self.base_results_dir = resolved_dir
        print(f"BiasAnalyzer initialized with results base directory: '{os.path.abspath(self.base_results_dir)}'")
        
    @property
    def embedder(self):
        if self._embedder is None and HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading sentence embedding model '{self.embedding_model_name}'...")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def locate_results_dir(self, model_id, dataset_id):
        """Attempts to find the results directory based on model and dataset names."""
        possible_dirs = [
            os.path.join(self.base_results_dir, f"{dataset_id}-{model_id}"),
            os.path.join(self.base_results_dir, f"{dataset_id}_{model_id}"),
            os.path.join(self.base_results_dir, f"{model_id}_{dataset_id}"),
            os.path.join(self.base_results_dir, f"{model_id}-{dataset_id}"),
            os.path.join(self.base_results_dir, model_id),
            # Also support variations of dataset name
            os.path.join(self.base_results_dir, f"truthfulqa-{model_id}" if "truth" in dataset_id.lower() else f"scienceqa-{model_id}"),
            os.path.join(self.base_results_dir, f"truthfulqa_{model_id}" if "truth" in dataset_id.lower() else f"scienceqa_{model_id}")
        ]
        
        for d in possible_dirs:
            if os.path.exists(d) and (glob.glob(os.path.join(d, "results_dict_*.pkl")) or glob.glob(os.path.join(d, "results_dict_*.pkl.gz"))):
                return d
                
        # If not found directly, check subfolders
        if os.path.exists(self.base_results_dir):
            for entry in os.listdir(self.base_results_dir):
                entry_path = os.path.join(self.base_results_dir, entry)
                if os.path.isdir(entry_path):
                    if dataset_id.lower() in entry.lower() and model_id.lower() in entry.lower():
                        return entry_path
                        
        raise FileNotFoundError(f"Could not find a results directory for model '{model_id}' and dataset '{dataset_id}' in '{self.base_results_dir}'.")

    def load_results_compiled(self, model_id, dataset_id):
        """Attempts to load pre-compiled results from entropies2 directory."""
        dataset_lower = dataset_id.lower()
        if 'truth' in dataset_lower:
            data = 'truth'
        elif 'open' in dataset_lower:
            data = 'open'
        elif 'science' in dataset_lower:
            data = 'scienceqa'
        else:
            data = dataset_lower

        modl = model_id.lower()

        # Resolve possible directories for entropies2
        candidate_dirs = [
            self.base_results_dir,
            os.path.join(self.base_results_dir, "entropies2"),
            os.path.join(os.path.dirname(self.base_results_dir), "entropies2"),
            os.path.join(os.path.dirname(os.path.dirname(self.base_results_dir)), "result", "entropies2"),
            os.path.join(self.base_results_dir, "..", "result", "entropies2")
        ]

        unique_dirs = []
        for d in candidate_dirs:
            if d and os.path.exists(d):
                abs_d = os.path.abspath(d)
                if abs_d not in unique_dirs:
                    unique_dirs.append(abs_d)

        for entropies2_dir in unique_dirs:
            for prefix in ["sdlg", "clar_sdlg"]:
                questions_file = os.path.join(entropies2_dir, f"{prefix}_question_{data}_{modl}_new.pkl")
                labels_file = os.path.join(entropies2_dir, f"{prefix}_label_{data}_{modl}_new.pkl")
                
                if os.path.exists(questions_file) and os.path.exists(labels_file):
                    try:
                        print(f"Fast-loading pre-compiled results from {questions_file} and {labels_file}...")
                        with open(questions_file, "rb") as f:
                            questions = pickle.load(f)
                        with open(labels_file, "rb") as f:
                            bleurt_scores = pickle.load(f)
                            
                        # If loaded list contains tensors/objects, extract scalar values safely
                        processed_bleurt = []
                        for score in bleurt_scores:
                            if hasattr(score, "item"):
                                processed_bleurt.append(score.item())
                            elif isinstance(score, (int, float)):
                                processed_bleurt.append(float(score))
                            else:
                                processed_bleurt.append(score)

                        df = pd.DataFrame({
                            "question_id": list(range(len(questions))),
                            "question": questions,
                            "best_answer": [""] * len(questions),
                            "greedy_answer": [""] * len(questions),
                            "bleurt": processed_bleurt
                        })
                        df = df.dropna(subset=["bleurt"])
                        return df
                    except Exception as e:
                        print(f"Warning: Failed to load pre-compiled results: {e}")
        return None

    def load_results(self, results_dir):
        """Loads all results_dict_*.pkl or .pkl.gz files from the results directory."""
        pkl_files = glob.glob(os.path.join(results_dir, "results_dict_*.pkl"))
        gz_files = glob.glob(os.path.join(results_dir, "results_dict_*.pkl.gz"))
        all_files = pkl_files + gz_files
        
        if not all_files:
            raise ValueError(f"No results_dict_*.pkl or .pkl.gz files found in {results_dir}")
            
        records = []
        for fpath in all_files:
            try:
                if fpath.endswith(".gz"):
                    with gzip.open(fpath, "rb") as f:
                        res = pickle.load(f)
                else:
                    with open(fpath, "rb") as f:
                        res = pickle.load(f)
                
                question = res.get("question", "")
                question_id = res.get("question_id", None)
                best_answer = res.get("best_answer", "")
                
                corr_dict = res.get("correctness_dict", {})
                bleurt_score = corr_dict.get("bleurt", None)
                
                greedy_answer = ""
                if "sdlg" in res and "generations" in res["sdlg"]:
                    gens = res["sdlg"]["generations"]
                    if gens and "generation_text" in gens[0]:
                        greedy_answer = gens[0]["generation_text"][0]
                elif "generations" in res:
                    gens = res["generations"]
                    if gens and "generation_text" in gens[0]:
                        greedy_answer = gens[0]["generation_text"][0]
                        
                records.append({
                    "question_id": question_id,
                    "question": question,
                    "best_answer": best_answer,
                    "greedy_answer": greedy_answer,
                    "bleurt": bleurt_score,
                })
            except Exception as e:
                pass
                
        df = pd.DataFrame(records)
        df = df.dropna(subset=["bleurt"])
        return df

    def fetch_dataset_metadata(self, df, dataset_name):
        """Loads dataset from Hugging Face to map questions to pre-defined categories."""
        if not HAS_DATASETS:
            return df
            
        try:
            if "truthful" in dataset_name.lower():
                ds = load_dataset("truthfulqa/truthful_qa", "generation")
                val_set = ds["validation"]
                lookup = {sample["question"].strip().lower(): sample.get("category", "Unknown") for sample in val_set}
                df["dataset_category"] = df["question"].apply(lambda q: lookup.get(q.strip().lower(), "Unknown"))
                
            elif "science" in dataset_name.lower():
                ds = load_dataset("tasksource/ScienceQA_text_only")
                test_set = ds["test"]
                
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
        except Exception as e:
            print(f"Warning: Could not fetch dataset metadata: {e}")
            
        return df

    def analyze(self, model_id, dataset_id, bleurt_threshold=-0.3, num_clusters=10, reduction_method="tsne", n_components=2, use_ground_truth=False, clustering_method="kmeans", dbscan_eps=0.5, dbscan_min_samples=5, cluster_on_projection=False, pca_dim_before_clustering=None):
        """Performs the complete analysis: loading, embedding, clustering, and dimension reduction."""
        # Try loading pre-compiled results first (very fast)
        df = self.load_results_compiled(model_id, dataset_id)
        
        if df is None:
            # Fallback to slow file-by-file loading
            print(f"Pre-compiled files for {model_id} on {dataset_id} not found in entropies2/. Falling back to slow file-by-file load...")
            results_dir = self.locate_results_dir(model_id, dataset_id)
            df = self.load_results(results_dir)
        
        if len(df) == 0:
            raise ValueError(f"No valid results with BLEURT scores found.")
            
        df = df.reset_index(drop=True)
        
        # 2. Fetch metadata categories
        df = self.fetch_dataset_metadata(df, dataset_id)
        
        # 3. Create question embeddings
        if self.embedder is None:
            raise ImportError("sentence-transformers is required for this analysis.")
        embeddings = self.embedder.encode(df["question"].tolist(), show_progress_bar=False)
        
        # 4. Perform clustering or group by ground truth
        has_valid_categories = False
        unique_cats = []
        if "dataset_category" in df.columns:
            unique_cats = df["dataset_category"].dropna().unique().tolist()
            valid_cats = [c for c in unique_cats if c and c != "Unknown"]
            if len(valid_cats) > 0:
                has_valid_categories = True
                
        if use_ground_truth and has_valid_categories:
            # Map each unique category to a cluster ID
            cat_to_id = {cat: idx for idx, cat in enumerate(unique_cats)}
            df["cluster"] = df["dataset_category"].map(cat_to_id)
            num_clusters = len(unique_cats)
            
            # Compute category centroids (mean embedding per category)
            category_centers = {}
            for cat in unique_cats:
                cat_indices = df[df["dataset_category"] == cat].index
                cat_embeddings = embeddings[cat_indices]
                category_centers[cat] = cat_embeddings.mean(axis=0)
                
            # Calculate distance of each question to its category center
            distances = []
            for i, row in df.iterrows():
                cat = row["dataset_category"]
                distances.append(np.linalg.norm(embeddings[i] - category_centers[cat]))
            df["dist_to_center"] = distances
            
            # Map topics directly to categories
            cluster_topics = {cat_to_id[cat]: f"{cat}" for cat in unique_cats}
            df["cluster_topic"] = df["dataset_category"]
            
            # Dimension reduction
            if reduction_method.lower() == "umap":
                if HAS_UMAP:
                    reducer = umap.UMAP(n_components=n_components, random_state=42)
                    coords = reducer.fit_transform(embeddings)
                else:
                    print("Warning: umap-learn package is not installed. Falling back to t-SNE.")
                    reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(df) - 1))
                    coords = reducer.fit_transform(embeddings)
            elif reduction_method.lower() == "tsne":
                reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(df) - 1))
                coords = reducer.fit_transform(embeddings)
            else:
                reducer = PCA(n_components=n_components, random_state=42)
                coords = reducer.fit_transform(embeddings)
            df["x"] = coords[:, 0]
            df["y"] = coords[:, 1]
            if n_components == 3:
                df["z"] = coords[:, 2]
        else:
            # Prepare clustering representation
            if cluster_on_projection:
                print(f"Projecting embeddings using {reduction_method} to {n_components}D before clustering...")
                if reduction_method.lower() == "umap":
                    if HAS_UMAP:
                        reducer = umap.UMAP(n_components=n_components, random_state=42)
                        coords = reducer.fit_transform(embeddings)
                    else:
                        print("Warning: umap-learn package is not installed. Falling back to t-SNE.")
                        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(df) - 1))
                        coords = reducer.fit_transform(embeddings)
                elif reduction_method.lower() == "tsne":
                    reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(df) - 1))
                    coords = reducer.fit_transform(embeddings)
                else:
                    reducer = PCA(n_components=n_components, random_state=42)
                    coords = reducer.fit_transform(embeddings)
                df["x"] = coords[:, 0]
                df["y"] = coords[:, 1]
                if n_components == 3:
                    df["z"] = coords[:, 2]
                clustering_embeddings = coords
            elif pca_dim_before_clustering is not None:
                print(f"Reducing embeddings to {pca_dim_before_clustering} PCA components before clustering...")
                pca_pre = PCA(n_components=pca_dim_before_clustering, random_state=42)
                clustering_embeddings = pca_pre.fit_transform(embeddings)
            else:
                clustering_embeddings = embeddings

            if num_clusters == "auto":
                if clustering_method.lower() == "dbscan":
                    # DBSCAN automatically finds its own number of clusters
                    pass
                else:
                    n_samples = len(clustering_embeddings)
                    k_min = 3
                    k_max = min(12, n_samples - 1)
                    
                    if k_max >= k_min:
                        print(f"Auto-clustering: searching optimal cluster count k in [{k_min}, {k_max}] using Silhouette Score...")
                        best_k = k_min
                        best_score = -1.0
                        for k in range(k_min, k_max + 1):
                            try:
                                if clustering_method.lower() == "agglomerative":
                                    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                                    labels = model.fit_predict(clustering_embeddings)
                                elif clustering_method.lower() == "gmm":
                                    model = GaussianMixture(n_components=k, random_state=42)
                                    labels = model.fit_predict(clustering_embeddings)
                                else:  # default is kmeans
                                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                                    labels = model.fit_predict(clustering_embeddings)
                                
                                n_unique = len(np.unique(labels))
                                if 1 < n_unique < n_samples:
                                    score = silhouette_score(clustering_embeddings, labels)
                                    if score > best_score:
                                        best_score = score
                                        best_k = k
                            except Exception as e:
                                print(f"Warning: Silhouette evaluation failed for k={k}: {e}")
                        num_clusters = best_k
                        print(f"Auto-clustering: selected optimal cluster count k = {num_clusters} (best silhouette score: {best_score:.4f})")
                    else:
                        num_clusters = min(3, max(2, n_samples - 1)) if n_samples > 2 else 2
                        print(f"Auto-clustering: too few samples ({n_samples}), using default cluster count: {num_clusters}")

            if clustering_method.lower() == "dbscan":
                dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                cluster_ids = dbscan.fit_predict(clustering_embeddings)
                df["cluster"] = cluster_ids
                
                # Determine number of clusters (excluding noise -1)
                unique_clusters = [c for c in np.unique(cluster_ids) if c != -1]
                num_clusters = len(unique_clusters)
                
                # Calculate cluster centroids manually
                cluster_centers = {}
                global_center = clustering_embeddings.mean(axis=0)
                for c_id in unique_clusters:
                    c_indices = df[df["cluster"] == c_id].index
                    cluster_centers[c_id] = clustering_embeddings[c_indices].mean(axis=0)
                cluster_centers[-1] = global_center
                
                # Calculate distance of each question to its cluster center
                distances = []
                for i, row in df.iterrows():
                    c_id = row["cluster"]
                    distances.append(np.linalg.norm(clustering_embeddings[i] - cluster_centers[c_id]))
                df["dist_to_center"] = distances
            elif clustering_method.lower() == "agglomerative":
                # Hierarchical Agglomerative Clustering using Ward's linkage
                agg = AgglomerativeClustering(n_clusters=num_clusters, linkage="ward")
                cluster_ids = agg.fit_predict(clustering_embeddings)
                df["cluster"] = cluster_ids
                
                # Calculate cluster centroids manually (mean embedding of each cluster)
                cluster_centers = {}
                for c_id in range(num_clusters):
                    c_indices = df[df["cluster"] == c_id].index
                    cluster_centers[c_id] = clustering_embeddings[c_indices].mean(axis=0)
                    
                # Calculate distance of each question to its cluster center
                distances = [np.linalg.norm(clustering_embeddings[i] - cluster_centers[c_id]) for i, c_id in enumerate(cluster_ids)]
                df["dist_to_center"] = distances
            elif clustering_method.lower() == "gmm":
                # Gaussian Mixture Model (probabilistic clustering)
                gmm = GaussianMixture(n_components=num_clusters, random_state=42)
                cluster_ids = gmm.fit_predict(clustering_embeddings)
                df["cluster"] = cluster_ids
                
                # Retrieve cluster centers (means)
                cluster_centers = gmm.means_
                
                # Calculate distance of each question to its cluster center
                distances = [np.linalg.norm(clustering_embeddings[i] - cluster_centers[c_id]) for i, c_id in enumerate(cluster_ids)]
                df["dist_to_center"] = distances
            else:
                # Unsupervised bottom-up KMeans clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                cluster_ids = kmeans.fit_predict(clustering_embeddings)
                df["cluster"] = cluster_ids
                
                # Calculate distances of questions to cluster centers
                cluster_centers = kmeans.cluster_centers_
                distances = [np.linalg.norm(clustering_embeddings[i] - cluster_centers[c_id]) for i, c_id in enumerate(cluster_ids)]
                df["dist_to_center"] = distances
            
            # Fit N-gram TF-IDF vectorizer on all questions to capture descriptive keyphrases
            try:
                phrase_vectorizer = TfidfVectorizer(
                    stop_words=CUSTOM_STOP_WORDS,
                    ngram_range=(1, 3),
                    max_features=5000,
                    min_df=1
                )
                phrase_vectorizer.fit(df["question"])
                feature_names = phrase_vectorizer.get_feature_names_out()
                has_vectorizer = True
            except Exception as e:
                print(f"Warning: Could not fit phrase vectorizer: {e}")
                has_vectorizer = False

            # Determine cluster topics using Two-Pass Descriptive N-gram Labeling
            cluster_topics = {}
            all_clusters = np.unique(cluster_ids).tolist()
            
            # Pass 1: Map cluster ID to its dominant category
            dominant_categories = {}
            category_counts = {}
            
            for c_id in all_clusters:
                if c_id == -1:
                    continue
                
                c_df = df[df["cluster"] == c_id]
                meta_summary = "Topic Group"
                if "dataset_category" in df.columns:
                    valid_cats = c_df["dataset_category"].dropna()
                    valid_cats = valid_cats[valid_cats != "Unknown"]
                    if len(valid_cats) > 0:
                        top_cat = valid_cats.value_counts().index
                        if len(top_cat) > 0:
                            meta_summary = top_cat[0]
                
                dominant_categories[c_id] = meta_summary
                category_counts[meta_summary] = category_counts.get(meta_summary, 0) + 1
            
            # Pass 2: Generate final descriptive labels
            group_indices = {}
            for c_id in all_clusters:
                if c_id == -1:
                    cluster_topics[-1] = "Outliers / Noise (C-1)"
                    continue
                
                cat = dominant_categories[c_id]
                count = category_counts[cat]
                
                if count > 1:
                    # Retrieve the most representative concept for this cluster
                    c_df = df[df["cluster"] == c_id]
                    
                    # 1. Topic metadata check (e.g. ScienceQA topic)
                    concept = ""
                    if "dataset_topic" in df.columns:
                        valid_topics = c_df["dataset_topic"].dropna()
                        valid_topics = valid_topics[valid_topics != "Unknown"]
                        if len(valid_topics) > 0:
                            top_topic = valid_topics.value_counts().index
                            if len(top_topic) > 0:
                                concept = top_topic[0]
                                
                    # 2. Keyphrase TF-IDF fallback (e.g. TruthfulQA or if topic is Unknown)
                    if (not concept or concept == "Unknown") and has_vectorizer:
                        try:
                            in_cluster_text = " ".join(c_df["question"])
                            cluster_tfidf = phrase_vectorizer.transform([in_cluster_text]).toarray()[0]
                            top_indices = np.argsort(cluster_tfidf)[::-1][:2]
                            top_phrases = [feature_names[idx] for idx in top_indices if cluster_tfidf[idx] > 0]
                            if len(top_phrases) > 0:
                                concept = " / ".join(top_phrases)
                        except Exception:
                            pass
                            
                    # Fallback to group index if no representative concept can be extracted
                    if not concept:
                        group_indices[cat] = group_indices.get(cat, 0) + 1
                        topic_label = f"{cat} - Group {group_indices[cat]} (C{c_id})"
                    else:
                        topic_label = f"{cat}: {concept} (C{c_id})"
                else:
                    topic_label = f"{cat} (C{c_id})"
                    
                cluster_topics[c_id] = topic_label
                
            df["cluster_topic"] = df["cluster"].map(cluster_topics)
            
            # If not clustered on projection, compute coords now
            if not cluster_on_projection:
                if reduction_method.lower() == "umap":
                    if HAS_UMAP:
                        reducer = umap.UMAP(n_components=n_components, random_state=42)
                        coords = reducer.fit_transform(embeddings)
                    else:
                        print("Warning: umap-learn package is not installed. Falling back to t-SNE.")
                        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(df) - 1))
                        coords = reducer.fit_transform(embeddings)
                elif reduction_method.lower() == "tsne":
                    reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(df) - 1))
                    coords = reducer.fit_transform(embeddings)
                else:
                    reducer = PCA(n_components=n_components, random_state=42)
                    coords = reducer.fit_transform(embeddings)
                df["x"] = coords[:, 0]
                df["y"] = coords[:, 1]
                if n_components == 3:
                    df["z"] = coords[:, 2]
        
        # 6. Correctness mapping
        df["correct"] = df["bleurt"] >= bleurt_threshold
        df["Correctness"] = df["correct"].map({True: "Correct", False: "Incorrect"})
        
        return {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "df": df,
            "cluster_topics": cluster_topics,
            "bleurt_threshold": bleurt_threshold
        }

    def plot_correctness(self, results, ax=None, figsize=(10, 8)):
        """Plots the 2D t-SNE scatter plot colored by correctness."""
        df = results["df"]
        model = results["model_id"]
        dataset = results["dataset_id"]
        thresh = results["bleurt_threshold"]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        sns.scatterplot(
            data=df, x="x", y="y", hue="Correctness",
            palette={"Correct": "#4CAF50", "Incorrect": "#F44336"},
            alpha=0.8, edgecolor="none", s=40, ax=ax
        )
        ax.set_title(f"{model} on {dataset}\nCorrectness Map (BLEURT >= {thresh})", fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc="upper right")
        
        if ax is None:
            plt.show()

    def plot_topics(self, results, ax=None, figsize=(10, 8)):
        """Plots the 2D t-SNE scatter plot colored by semantic topics."""
        df = results["df"]
        model = results["model_id"]
        dataset = results["dataset_id"]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        sns.scatterplot(
            data=df, x="x", y="y", hue="cluster_topic",
            palette="tab10", alpha=0.8, edgecolor="none", s=40, ax=ax
        )
        ax.set_title(f"{model} on {dataset}\nTopic Clusters Map", fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Topics")
        
        if ax is None:
            plt.show()

    def print_bias_summary(self, results):
        """Prints a clean Markdown table summarizing performance by topic cluster."""
        df = results["df"]
        model = results["model_id"]
        dataset = results["dataset_id"]
        thresh = results["bleurt_threshold"]
        
        summary_data = []
        grouped = df.groupby("cluster")
        for c_id, group in grouped:
            total = len(group)
            correct = group["correct"].sum()
            accuracy = (correct / total) * 100
            avg_bleurt = group["bleurt"].mean()
            topic = group["cluster_topic"].iloc[0]
            
            # Top representative questions
            rep_questions = group.nsmallest(2, "dist_to_center")["question"].tolist()
            rep_str = " | ".join([f'"{q[:60]}..."' for q in rep_questions])
            
            summary_data.append({
                "Topic": topic,
                "Total Qs": total,
                "Accuracy (%)": f"{accuracy:.1f}%",
                "Avg BLEURT": f"{avg_bleurt:.3f}",
                "Representative Questions": rep_str,
                "raw_acc": accuracy
            })
            
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by="raw_acc")  # Show worst performing first
        summary_df = summary_df.drop(columns=["raw_acc"])
        
        overall_acc = df["correct"].mean() * 100
        
        md_text = f"### Topic Bias Analysis: **{model}** on **{dataset}**\n"
        md_text += f"* **Overall Accuracy**: `{overall_acc:.2f}%` (Threshold: BLEURT $\\ge$ `{thresh}`)\n"
        md_text += f"* **Total Questions**: `{len(df)}`\n\n"
        md_text += "| Topic / Cluster | Total Qs | Accuracy (%) | Avg BLEURT | Representative Questions |\n"
        md_text += "| :--- | :---: | :---: | :---: | :--- |\n"
        
        for _, row in summary_df.iterrows():
            md_text += f"| {row['Topic']} | {row['Total Qs']} | **{row['Accuracy (%)']}** | {row['Avg BLEURT']} | {row['Representative Questions']} |\n"
            
        if HAS_IPYTHON:
            display(Markdown(md_text))
        else:
            print(md_text)

    def plot_interactive_plotly(self, results, color_by="cluster_topic", show_plot=True, save_html_path=None):
        """Generates an interactive Plotly scatter plot for notebook view."""
        if not HAS_PLOTLY:
            print("Plotly is not installed. Run `pip install plotly` for interactive dashboards.")
            return None
            
        df = results["df"]
        model = results["model_id"]
        dataset = results["dataset_id"]
        thresh = results["bleurt_threshold"]
        
        # Configure renderer for remote Jupyter environments
        try:
            import plotly.io as pio
            pio.renderers.default = "notebook_connected"
        except Exception:
            pass
            
        # Hover info columns
        hover_data = ["question", "best_answer", "greedy_answer", "bleurt", "cluster_topic"]
        if "dataset_category" in df.columns:
            hover_data.append("dataset_category")
            
        # Create a copy to prevent side-effects on the cached dataframe
        df = df.copy()
        
        # Calculate cluster accuracy percentages for legend labels
        cluster_accs = {}
        if "cluster" in df.columns and "correct" in df.columns:
            grouped = df.groupby("cluster")
            for c_id, group in grouped:
                total_qs = len(group)
                correct_qs = group["correct"].sum()
                accuracy = (correct_qs / total_qs) * 100
                topic = group["cluster_topic"].iloc[0]
                cluster_accs[topic] = accuracy

        # We will map "Correctness" to simple "Correct" / "Incorrect"
        df["Correctness"] = df["correct"].map({True: "Correct", False: "Incorrect"})
        
        # Configure dynamic colors and symbols
        color_args = {}
        symbol_args = {}
        
        if color_by == "Correctness":
            color_args = {
                "color": "Correctness",
                "color_discrete_map": {"Correct": "#10B981", "Incorrect": "#F43F5E"}
            }
        else:
            c_col = color_by if color_by in df.columns else "cluster_topic"
            color_args = {
                "color": c_col,
                "color_discrete_sequence": px.colors.qualitative.Prism
            }
            if "Correctness" in df.columns:
                symbol_args = {
                    "symbol": "Correctness",
                    "symbol_map": {"Correct": "circle", "Incorrect": "diamond"}
                }
        
        if "z" in df.columns:
            # 3D Plotly Scatter Plot
            fig = px.scatter_3d(
                df, x="x", y="y", z="z",
                hover_name="question_id",
                hover_data={col: True for col in hover_data},
                title=f"3D Semantic Correctness Space: {model} on {dataset}",
                **color_args,
                **symbol_args
            )
            
            # Group legend by color to avoid combined legend items
            added_clusters = set()
            for trace in fig.data:
                name_parts = trace.name.split(", ")
                if len(name_parts) == 2:
                    cluster_name, status = name_parts
                    acc = cluster_accs.get(cluster_name, None)
                    
                    if cluster_name not in added_clusters:
                        if acc is not None:
                            trace.name = f"{cluster_name} ({acc:.1f}%)"
                        else:
                            trace.name = cluster_name
                        trace.showlegend = True
                        added_clusters.add(cluster_name)
                    else:
                        trace.showlegend = False
                        
            # Add shape legend at the bottom using graph_objects
            import plotly.graph_objects as go
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(symbol="circle", color="#94a3b8", size=5),
                name="Correct Answer",
                showlegend=True
            ))
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(symbol="diamond", color="#94a3b8", size=5),
                name="Incorrect Answer",
                showlegend=True
            ))
            
            # Premium dark dashboard design
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                scene=dict(
                    xaxis=dict(gridcolor="#334155", backgroundcolor="#0f172a", showbackground=True, title="t-SNE Dim 1"),
                    yaxis=dict(gridcolor="#334155", backgroundcolor="#0f172a", showbackground=True, title="t-SNE Dim 2"),
                    zaxis=dict(gridcolor="#334155", backgroundcolor="#0f172a", showbackground=True, title="t-SNE Dim 3")
                ),
                title=dict(
                    text=f"<b>3D Semantic Space & Correctness Map</b><br><sup>Model: {model} | Dataset: {dataset} (BLEURT >= {thresh}) | Color: {color_by} (Accuracy %) | Shape: Correctness</sup>",
                    font=dict(family="System-UI, sans-serif", size=18, color="#f8fafc"),
                    x=0.05,
                    y=0.95
                ),
                legend=dict(
                    title="Legend",
                    font=dict(color="#f8fafc"),
                    bgcolor="rgba(15, 23, 42, 0.8)",
                    bordercolor="#334155",
                    borderwidth=1,
                    yanchor="top",
                    y=0.9,
                    xanchor="left",
                    x=0.05
                ),
                width=1000,
                height=800
            )
            
            fig.update_traces(
                marker=dict(
                    size=4.5,
                    opacity=0.85,
                    line=dict(width=0.5, color="rgba(255,255,255,0.4)")
                )
            )
        else:
            # 2D Plotly Scatter Plot
            fig = px.scatter(
                df, x="x", y="y", 
                hover_name="question_id",
                hover_data={col: True for col in hover_data},
                title=f"2D Semantic Correctness Space: {model} on {dataset}",
                **color_args,
                **symbol_args
            )
            
            # Group legend by color to avoid combined legend items
            added_clusters = set()
            for trace in fig.data:
                name_parts = trace.name.split(", ")
                if len(name_parts) == 2:
                    cluster_name, status = name_parts
                    acc = cluster_accs.get(cluster_name, None)
                    
                    if cluster_name not in added_clusters:
                        if acc is not None:
                            trace.name = f"{cluster_name} ({acc:.1f}%)"
                        else:
                            trace.name = cluster_name
                        trace.showlegend = True
                        added_clusters.add(cluster_name)
                    else:
                        trace.showlegend = False
                        
            # Add shape legend at the bottom using graph_objects
            import plotly.graph_objects as go
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="circle", color="#94a3b8", size=8),
                name="Correct Answer",
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="diamond", color="#94a3b8", size=8),
                name="Incorrect Answer",
                showlegend=True
            ))
            
            # Premium dark dashboard design
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                xaxis=dict(showgrid=True, gridcolor="#334155", title="t-SNE Dim 1"),
                yaxis=dict(showgrid=True, gridcolor="#334155", title="t-SNE Dim 2"),
                title=dict(
                    text=f"<b>2D Semantic Space & Correctness Map</b><br><sup>Model: {model} | Dataset: {dataset} (BLEURT >= {thresh}) | Color: {color_by} (Accuracy %) | Shape: Correctness</sup>",
                    font=dict(family="System-UI, sans-serif", size=18, color="#f8fafc"),
                    x=0.05,
                    y=0.95
                ),
                legend=dict(
                    title="Legend",
                    font=dict(color="#f8fafc"),
                    bgcolor="rgba(15, 23, 42, 0.8)",
                    bordercolor="#334155",
                    borderwidth=1
                ),
                width=900,
                height=600
            )
            
            fig.update_traces(
                marker=dict(
                    size=8,
                    opacity=0.85,
                    line=dict(width=0.8, color="rgba(255,255,255,0.4)")
                )
            )
        
        if save_html_path is None:
            clean_model = model.replace("/", "_").replace("\\", "_")
            save_html_path = f"plotly_interactive_{clean_model}_{dataset}.html"
            
        try:
            fig.write_html(save_html_path)
            print(f"✓ Saved interactive HTML dashboard to: {os.path.abspath(save_html_path)}")
            if HAS_IPYTHON:
                from IPython.display import display, HTML
                display(HTML(f'<div style="margin: 10px 0; padding: 10px; background-color: #0f172a; border-left: 5px solid #10b981; border-radius: 4px;">'
                             f'<span style="color: #10b981; font-weight: bold;">Interactive Dashboard Saved:</span> '
                             f'<span style="color: #f8fafc;">If the plot does not display inline above, click the link to open it in a new tab:</span> '
                             f'<a href="{save_html_path}" target="_blank" style="font-weight: bold; text-decoration: underline; color: #38bdf8;">'
                             f'Open Dashboard ({save_html_path})</a></div>'))
        except Exception as e:
            print(f"Warning: Could not save HTML file: {e}")
            
        if show_plot:
            try:
                fig.show()
            except Exception as e:
                print(f"Note: Inline Plotly display failed (common in remote Jupyter environments). Use the HTML link above to open it.")
            
        return fig


    def plot_bleurt_histogram(self, model_id, dataset_id, bins=30, ax=None, figsize=(10, 6)):
        """Plots a histogram of BLEURT scores for a model and dataset to help calibrate thresholds."""
        df = self.load_results_compiled(model_id, dataset_id)
        if df is None:
            try:
                print(f"Pre-compiled files for {model_id} on {dataset_id} not found in entropies2/. Falling back to slow file-by-file load...")
                results_dir = self.locate_results_dir(model_id, dataset_id)
                df = self.load_results(results_dir)
            except Exception as e:
                print(f"Could not load results for {model_id} on {dataset_id}: {e}")
                return
            
        if len(df) == 0 or "bleurt" not in df.columns:
            print(f"No valid BLEURT scores found for {model_id} on {dataset_id}.")
            return
            
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        sns.histplot(df["bleurt"], bins=bins, kde=True, color="skyblue", ax=ax, edgecolor="black")
        
        # Calculate key statistics
        mean_score = df["bleurt"].mean()
        median_score = df["bleurt"].median()
        
        ax.axvline(mean_score, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_score:.3f}")
        ax.axvline(median_score, color="green", linestyle="-.", linewidth=1.5, label=f"Median: {median_score:.3f}")
        
        ax.set_title(f"BLEURT Score Distribution: {model_id} on {dataset_id}", fontsize=12, fontweight="bold")
        ax.set_xlabel("BLEURT Score")
        ax.set_ylabel("Count")
        ax.legend()
        
        if ax is None:
            plt.show()

    def plot_dbscan_k_distance(self, model_id, dataset_id, k=5, ax=None, figsize=(10, 6)):
        """Plots the k-distance graph to help determine the optimal 'eps' for DBSCAN."""
        df = self.load_results_compiled(model_id, dataset_id)
        if df is None:
            try:
                results_dir = self.locate_results_dir(model_id, dataset_id)
                df = self.load_results(results_dir)
            except Exception as e:
                raise ValueError(f"Could not load results for model {model_id} on {dataset_id}: {e}")
            
        df = df.reset_index(drop=True)
        if self.embedder is None:
            raise ImportError("sentence-transformers is required to generate embeddings.")
        embeddings = self.embedder.encode(df["question"].tolist(), show_progress_bar=False)
        
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Sort distances to the k-th nearest neighbor (k-th index is columns k-1)
        k_distances = np.sort(distances[:, k-1])[::-1]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        ax.plot(k_distances, color="#38bdf8", lw=2)
        ax.set_title(f"DBSCAN K-Distance Graph (k={k}) for {model_id} on {dataset_id}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Points sorted by distance (descending)")
        ax.set_ylabel(f"{k}-th Nearest Neighbor Distance (eps)")
        ax.grid(True, linestyle=":", alpha=0.5)
        
        # Highlight typical ranges
        ax.axhspan(0.4, 0.6, color="yellow", alpha=0.15, label="Typical eps range (0.4 - 0.6)")
        ax.legend()
        
        if ax is None:
            plt.show()

    def analyze_comparative(self, model_ids, dataset_id, bleurt_threshold=-0.3, num_clusters=12, clustering_method="kmeans", dbscan_eps=0.5, dbscan_min_samples=5, cluster_on_projection=False, pca_dim_before_clustering=None, reduction_method="tsne"):
        """Runs clustering on shared questions once, then computes accuracy for all models on those clusters."""
        # 1. Load questions from the first model
        df_first = self.load_results_compiled(model_ids[0], dataset_id)
        if df_first is None:
            try:
                results_dir = self.locate_results_dir(model_ids[0], dataset_id)
                df_first = self.load_results(results_dir)
            except Exception as e:
                raise ValueError(f"Could not load results for first model {model_ids[0]} to establish clusters: {e}")
                
        questions = df_first["question"].tolist()
        
        # 2. Embed and cluster once to define shared topics
        if self.embedder is None:
            raise ImportError("sentence-transformers is required for clustering.")
        
        print("Generating shared question embeddings...")
        embeddings = self.embedder.encode(questions, show_progress_bar=False)
        
        # Prepare clustering representation
        if cluster_on_projection:
            print("Reducing embeddings to 2D projection before clustering...")
            if reduction_method.lower() == "umap":
                if HAS_UMAP:
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    clustering_embeddings = reducer.fit_transform(embeddings)
                else:
                    print("Warning: umap-learn package is not installed. Falling back to t-SNE.")
                    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(questions) - 1))
                    clustering_embeddings = reducer.fit_transform(embeddings)
            elif reduction_method.lower() == "tsne":
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(questions) - 1))
                clustering_embeddings = reducer.fit_transform(embeddings)
            else:
                reducer = PCA(n_components=2, random_state=42)
                clustering_embeddings = reducer.fit_transform(embeddings)
        elif pca_dim_before_clustering is not None:
            print(f"Reducing embeddings to {pca_dim_before_clustering} PCA components before clustering...")
            reducer = PCA(n_components=pca_dim_before_clustering, random_state=42)
            clustering_embeddings = reducer.fit_transform(embeddings)
        else:
            clustering_embeddings = embeddings
        
        if num_clusters == "auto":
            if clustering_method.lower() == "dbscan":
                # DBSCAN automatically finds its own number of clusters
                pass
            else:
                n_samples = len(clustering_embeddings)
                k_min = 3
                k_max = min(12, n_samples - 1)
                
                if k_max >= k_min:
                    print(f"Auto-clustering: searching optimal cluster count k in [{k_min}, {k_max}] using Silhouette Score...")
                    best_k = k_min
                    best_score = -1.0
                    for k in range(k_min, k_max + 1):
                        try:
                            if clustering_method.lower() == "agglomerative":
                                model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                                labels = model.fit_predict(clustering_embeddings)
                            elif clustering_method.lower() == "gmm":
                                model = GaussianMixture(n_components=k, random_state=42)
                                labels = model.fit_predict(clustering_embeddings)
                            else:  # default is kmeans
                                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                                labels = model.fit_predict(clustering_embeddings)
                            
                            n_unique = len(np.unique(labels))
                            if 1 < n_unique < n_samples:
                                score = silhouette_score(clustering_embeddings, labels)
                                if score > best_score:
                                    best_score = score
                                    best_k = k
                        except Exception as e:
                            print(f"Warning: Silhouette evaluation failed for k={k}: {e}")
                    num_clusters = best_k
                    print(f"Auto-clustering: selected optimal cluster count k = {num_clusters} (best silhouette score: {best_score:.4f})")
                else:
                    num_clusters = min(3, max(2, n_samples - 1)) if n_samples > 2 else 2
                    print(f"Auto-clustering: too few samples ({n_samples}), using default cluster count: {num_clusters}")

        if clustering_method.lower() == "dbscan":
            print(f"Clustering shared questions using DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples})...")
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            cluster_ids = dbscan.fit_predict(clustering_embeddings)
            
            # Determine unique clusters (excluding noise -1)
            unique_clusters = [c for c in np.unique(cluster_ids) if c != -1]
            num_clusters = len(unique_clusters)
            
            # Calculate cluster centroids manually
            cluster_centers = {}
            global_center = clustering_embeddings.mean(axis=0)
            for c_id in unique_clusters:
                c_indices = np.where(cluster_ids == c_id)[0]
                cluster_centers[c_id] = clustering_embeddings[c_indices].mean(axis=0)
            cluster_centers[-1] = global_center
            
            # Calculate distance of each question to its cluster center
            distances = []
            for i, c_id in enumerate(cluster_ids):
                distances.append(np.linalg.norm(clustering_embeddings[i] - cluster_centers[c_id]))
        elif clustering_method.lower() == "agglomerative":
            print(f"Clustering shared questions into {num_clusters} topics using Agglomerative Clustering...")
            agg = AgglomerativeClustering(n_clusters=num_clusters, linkage="ward")
            cluster_ids = agg.fit_predict(clustering_embeddings)
            
            # Calculate cluster centroids manually
            cluster_centers = {}
            for c_id in range(num_clusters):
                c_indices = np.where(cluster_ids == c_id)[0]
                cluster_centers[c_id] = clustering_embeddings[c_indices].mean(axis=0)
                
            # Calculate distance of each question to its cluster center
            distances = [np.linalg.norm(clustering_embeddings[i] - cluster_centers[c_id]) for i, c_id in enumerate(cluster_ids)]
        elif clustering_method.lower() == "gmm":
            print(f"Clustering shared questions into {num_clusters} topics using GMM...")
            gmm = GaussianMixture(n_components=num_clusters, random_state=42)
            cluster_ids = gmm.fit_predict(clustering_embeddings)
            
            # Retrieve cluster centers
            cluster_centers = gmm.means_
            
            # Calculate distance of each question to its cluster center
            distances = [np.linalg.norm(clustering_embeddings[i] - cluster_centers[c_id]) for i, c_id in enumerate(cluster_ids)]
        else:
            print(f"Clustering shared questions into {num_clusters} topics using KMeans...")
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(clustering_embeddings)
            
            # Calculate centroids and label clusters
            cluster_centers = kmeans.cluster_centers_
            distances = [np.linalg.norm(clustering_embeddings[i] - cluster_centers[c_id]) for i, c_id in enumerate(cluster_ids)]
            
        # Create temp DataFrame for metadata mapping and cluster labeling
        temp_df = pd.DataFrame({
            "question": questions,
            "cluster": cluster_ids,
            "dist_to_center": distances
        })
        temp_df = self.fetch_dataset_metadata(temp_df, dataset_id)
        
        # Fit N-gram TF-IDF vectorizer on all questions to capture descriptive keyphrases
        try:
            phrase_vectorizer = TfidfVectorizer(
                stop_words=CUSTOM_STOP_WORDS,
                ngram_range=(1, 3),
                max_features=5000,
                min_df=1
            )
            phrase_vectorizer.fit(questions)
            feature_names = phrase_vectorizer.get_feature_names_out()
            has_vectorizer = True
        except Exception as e:
            print(f"Warning: Could not fit phrase vectorizer: {e}")
            has_vectorizer = False

        # Determine cluster topics using Two-Pass Descriptive N-gram Labeling
        cluster_topics = {}
        all_clusters = np.unique(cluster_ids).tolist()
        
        # Pass 1: Map cluster ID to its dominant category
        dominant_categories = {}
        category_counts = {}
        
        for c_id in all_clusters:
            if c_id == -1:
                continue
            
            c_df = temp_df[temp_df["cluster"] == c_id]
            meta_summary = "Topic Group"
            if "dataset_category" in temp_df.columns:
                valid_cats = c_df["dataset_category"].dropna()
                valid_cats = valid_cats[valid_cats != "Unknown"]
                if len(valid_cats) > 0:
                    top_cat = valid_cats.value_counts().index
                    if len(top_cat) > 0:
                        meta_summary = top_cat[0]
            
            dominant_categories[c_id] = meta_summary
            category_counts[meta_summary] = category_counts.get(meta_summary, 0) + 1
        
        # Pass 2: Generate final descriptive labels
        group_indices = {}
        for c_id in all_clusters:
            if c_id == -1:
                cluster_topics[-1] = "Outliers / Noise (C-1)"
                continue
            
            cat = dominant_categories[c_id]
            count = category_counts[cat]
            
            if count > 1:
                # Retrieve the most representative concept for this cluster
                c_df = temp_df[temp_df["cluster"] == c_id]
                
                # 1. Topic metadata check (e.g. ScienceQA topic)
                concept = ""
                if "dataset_topic" in temp_df.columns:
                    valid_topics = c_df["dataset_topic"].dropna()
                    valid_topics = valid_topics[valid_topics != "Unknown"]
                    if len(valid_topics) > 0:
                        top_topic = valid_topics.value_counts().index
                        if len(top_topic) > 0:
                            concept = top_topic[0]
                            
                # 2. Keyphrase TF-IDF fallback (e.g. TruthfulQA or if topic is Unknown)
                if (not concept or concept == "Unknown") and has_vectorizer:
                    try:
                        in_cluster_text = " ".join(c_df["question"])
                        cluster_tfidf = phrase_vectorizer.transform([in_cluster_text]).toarray()[0]
                        top_indices = np.argsort(cluster_tfidf)[::-1][:2]
                        top_phrases = [feature_names[idx] for idx in top_indices if cluster_tfidf[idx] > 0]
                        if len(top_phrases) > 0:
                            concept = " / ".join(top_phrases)
                    except Exception:
                        pass
                        
                # Fallback to group index if no representative concept can be extracted
                if not concept:
                    group_indices[cat] = group_indices.get(cat, 0) + 1
                    topic_label = f"{cat} - Group {group_indices[cat]} (C{c_id})"
                else:
                    topic_label = f"{cat}: {concept} (C{c_id})"
            else:
                topic_label = f"{cat} (C{c_id})"
            cluster_topics[c_id] = topic_label

        temp_df["cluster_topic"] = temp_df["cluster"].map(cluster_topics)
        
        # 3. Load BLEURT scores for each model and evaluate accuracy on the shared clusters
        comparison_records = []
        for model in model_ids:
            print(f"Loading correctness labels for model: {model}...")
            df_model = self.load_results_compiled(model, dataset_id)
            if df_model is None:
                try:
                    results_dir = self.locate_results_dir(model, dataset_id)
                    df_model = self.load_results(results_dir)
                except Exception as e:
                    print(f"Warning: Skipping model {model} due to load error: {e}")
                    continue
            
            # Map BLEURT scores to the questions
            score_lookup = {row["question"].strip().lower(): row["bleurt"] for _, row in df_model.iterrows()}
            
            # Calculate accuracy per cluster for this model
            for c_id in all_clusters:
                c_df = temp_df[temp_df["cluster"] == c_id]
                scores = [score_lookup.get(q.strip().lower(), None) for q in c_df["question"]]
                valid_scores = [s for s in scores if s is not None]
                
                if len(valid_scores) > 0:
                    correct_count = sum(1 for s in valid_scores if s >= bleurt_threshold)
                    accuracy = (correct_count / len(valid_scores)) * 100
                else:
                    accuracy = np.nan
                    
                comparison_records.append({
                    "model": model,
                    "cluster_id": c_id,
                    "topic_label": cluster_topics[c_id],
                    "accuracy": accuracy,
                    "num_questions": len(c_df)
                })
                
        comparison_df = pd.DataFrame(comparison_records)
        return {
            "comparison_df": comparison_df,
            "dataset_id": dataset_id,
            "bleurt_threshold": bleurt_threshold,
            "cluster_topics": cluster_topics
        }

    def plot_comparative_heatmap(self, comparative_results, ax=None, figsize=(12, 10)):
        """Plots a beautiful heatmap comparing model accuracy across shared topic clusters."""
        df = comparative_results["comparison_df"]
        dataset = comparative_results["dataset_id"]
        thresh = comparative_results["bleurt_threshold"]
        
        # Pivot the dataframe to get Topics as rows and Models as columns
        pivot_df = df.pivot(index="topic_label", columns="model", values="accuracy")
        
        # Reorder index to show clusters with lowest average accuracy first (worst topics first)
        avg_acc = pivot_df.mean(axis=1)
        pivot_df = pivot_df.loc[avg_acc.sort_values().index]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        sns.heatmap(
            pivot_df, annot=True, fmt=".1f", cmap="RdYlGn", 
            linewidths=0.5, cbar_kws={'label': 'Accuracy (%)'}, ax=ax,
            vmin=0, vmax=100
        )
        
        ax.set_title(f"Model Accuracy Comparison by Topic Cluster\nDataset: {dataset} (BLEURT >= {thresh})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Language Models", fontsize=11)
        ax.set_ylabel("Shared Topic Clusters (Sorted: Worst to Best)", fontsize=11)
        plt.tight_layout()
        
        if ax is None:
            plt.show()

    def plot_comparative_bar(self, comparative_results, ax=None, figsize=(12, 10)):
        """Plots a grouped horizontal bar chart comparing model accuracy across topic clusters."""
        df = comparative_results["comparison_df"]
        dataset = comparative_results["dataset_id"]
        thresh = comparative_results["bleurt_threshold"]
        
        # Calculate overall difficulty of each topic (mean accuracy across models)
        # to sort the topics on the Y-axis
        topic_order = df.groupby("topic_label")["accuracy"].mean().sort_values().index
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        sns.barplot(
            data=df, y="topic_label", x="accuracy", hue="model",
            order=topic_order, palette="muted", ax=ax, edgecolor="black", alpha=0.9
        )
        
        ax.set_title(f"Model Performance Comparison by Topic Cluster\nDataset: {dataset} (BLEURT >= {thresh})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Accuracy (%)", fontsize=11)
        ax.set_ylabel("Topic Clusters (Sorted: Hardest to Easiest)", fontsize=11)
        ax.set_xlim(0, 100)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Draw a vertical line at 50% accuracy for reference
        ax.axvline(50, color='grey', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        if ax is None:
            plt.show()

    def load_detailed_results(self, model_id, dataset_id):
        """Loads detailed results (including answers and generations) by scanning individual pickles."""
        results_dir = self.locate_results_dir(model_id, dataset_id)
        df = self.load_results(results_dir)
        return df

    def analyze_comparative_ground_truth(self, model_ids, dataset_id, bleurt_threshold=-0.3, group_by_column="dataset_category"):
        """Runs comparative analysis using the dataset's human-annotated categories directly, instead of clustering."""
        # 1. Load questions and metadata
        df_first = self.load_results_compiled(model_ids[0], dataset_id)
        if df_first is None:
            try:
                results_dir = self.locate_results_dir(model_ids[0], dataset_id)
                df_first = self.load_results(results_dir)
            except Exception as e:
                raise ValueError(f"Could not load results for first model {model_ids[0]}: {e}")
                
        # Fetch metadata categories
        df_first = self.fetch_dataset_metadata(df_first, dataset_id)
        
        # Verify the requested column exists
        if group_by_column not in df_first.columns:
            available_cols = [col for col in ["dataset_category", "dataset_topic", "dataset_subject"] if col in df_first.columns]
            if available_cols:
                group_by_column = available_cols[0]
            else:
                raise ValueError(f"No metadata category columns found in the dataset.")
                
        # Get unique categories and filter out very small categories (fewer than 2 questions)
        category_counts = df_first[group_by_column].value_counts()
        valid_categories = category_counts[category_counts >= 2].index.tolist()
        
        # 2. Load BLEURT scores for each model and evaluate accuracy on these categories
        comparison_records = []
        for model in model_ids:
            df_model = self.load_results_compiled(model, dataset_id)
            if df_model is None:
                try:
                    results_dir = self.locate_results_dir(model, dataset_id)
                    df_model = self.load_results(results_dir)
                except Exception as e:
                    print(f"Warning: Skipping model {model} due to load error: {e}")
                    continue
            
            # Map BLEURT scores to the questions
            score_lookup = {row["question"].strip().lower(): row["bleurt"] for _, row in df_model.iterrows()}
            
            for cat in valid_categories:
                cat_questions = df_first[df_first[group_by_column] == cat]["question"].tolist()
                scores = [score_lookup.get(q.strip().lower(), None) for q in cat_questions]
                valid_scores = [s for s in scores if s is not None]
                
                if len(valid_scores) > 0:
                    correct_count = sum(1 for s in valid_scores if s >= bleurt_threshold)
                    accuracy = (correct_count / len(valid_scores)) * 100
                else:
                    accuracy = np.nan
                    
                comparison_records.append({
                    "model": model,
                    "topic_label": cat,
                    "accuracy": accuracy,
                    "num_questions": len(valid_scores)
                })
                
        comparison_df = pd.DataFrame(comparison_records)
        return {
            "comparison_df": comparison_df,
            "dataset_id": dataset_id,
            "bleurt_threshold": bleurt_threshold,
            "group_by_column": group_by_column
        }

    def plot_topic_centroid_map(self, results, ax=None, figsize=(11, 10)):
        """Plots an aggregated topic centroid map. Size of circle represents question count,
        and color represents accuracy (red to green).
        """
        df = results["df"]
        model = results["model_id"]
        dataset = results["dataset_id"]
        
        # Group by cluster to calculate centroid positions, accuracy, and size
        grouped = df.groupby("cluster")
        centroid_data = []
        for c_id, group in grouped:
            x_mean = group["x"].mean()
            y_mean = group["y"].mean()
            
            total_qs = len(group)
            correct_qs = group["correct"].sum()
            accuracy = (correct_qs / total_qs) * 100
            topic = group["cluster_topic"].iloc[0]
            
            centroid_data.append({
                "cluster_id": c_id,
                "topic_label": topic,
                "x": x_mean,
                "y": y_mean,
                "accuracy": accuracy,
                "count": total_qs
            })
            
        c_df = pd.DataFrame(centroid_data)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        sizes = c_df["count"] * 60
        
        scatter = ax.scatter(
            c_df["x"], c_df["y"], s=sizes, c=c_df["accuracy"],
            cmap="RdYlGn", edgecolor="black", alpha=0.85, vmin=0, vmax=100
        )
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label("Accuracy (%)", fontsize=10)
        
        # Add labels directly on top of the circles
        for _, row in c_df.iterrows():
            label_text = f"{row['topic_label']}\n({row['count']} Qs)\n{row['accuracy']:.1f}%"
            ax.text(
                row["x"], row["y"], label_text,
                ha="center", va="center", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="none")
            )
            
        x_padding = (c_df["x"].max() - c_df["x"].min()) * 0.15
        y_padding = (c_df["y"].max() - c_df["y"].min()) * 0.15
        ax.set_xlim(c_df["x"].min() - x_padding, c_df["x"].max() + x_padding)
        ax.set_ylim(c_df["y"].min() - y_padding, c_df["y"].max() + y_padding)
        
        ax.set_title(f"Semantic Topic Accuracy Map: {model} on {dataset}\nCircle size = Q count | Color = Accuracy", fontsize=12, fontweight="bold")
        ax.set_xlabel("Semantic Dimension 1")
        ax.set_ylabel("Semantic Dimension 2")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.5)
        
        plt.tight_layout()
        if ax is None:
            plt.show()

    def plot_accuracy_landscape(self, results, ax=None, figsize=(11, 9)):
        """Plots a continuous semantic accuracy landscape using interpolation/contours.
        Shows areas of high and low correctness as peaks and valleys in the semantic space.
        """
        df = results["df"]
        model = results["model_id"]
        dataset = results["dataset_id"]
        
        from scipy.interpolate import griddata
        
        # Create a grid
        grid_x, grid_y = np.mgrid[
            df["x"].min():df["x"].max():100j,
            df["y"].min():df["y"].max():100j
        ]
        
        points = df[["x", "y"]].values
        values = df["correct"].astype(float).values * 100
        
        try:
            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
            grid_z = np.nan_to_num(grid_z, nan=values.mean())
        except Exception:
            grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')
            
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        contour = ax.contourf(
            grid_x, grid_y, grid_z, levels=15, cmap="RdYlGn", alpha=0.75,
            vmin=0, vmax=100
        )
        
        # Overlay the topic cluster names at their centroid positions
        grouped = df.groupby("cluster")
        for c_id, group in grouped:
            ax.text(
                group["x"].mean(), group["y"].mean(), group["cluster_topic"].iloc[0],
                ha="center", va="center", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="grey", lw=0.5)
            )
            
        cbar = plt.colorbar(contour, ax=ax, shrink=0.7)
        cbar.set_label("Accuracy Surface Density (%)", fontsize=10)
        
        ax.set_title(f"Factual Correctness Landscape: {model} on {dataset}\nRed valleys = Model Biases | Green hills = Strong areas", fontsize=12, fontweight="bold")
        ax.set_xlabel("Semantic Dimension 1")
        ax.set_ylabel("Semantic Dimension 2")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        if ax is None:
            plt.show()
