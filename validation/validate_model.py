"""
Validation Script for OASIS Teeth Similarity Search Model

This script evaluates the model's ability to find the original tooth scan
when given an augmented (third) view. The validation process:

1. Load the trained model
2. Generate embeddings for all original tooth scans (gallery)
3. Generate embeddings for all third-view augmented scans (queries)
4. For each query, find the most similar scans in the gallery
5. Check if the original scan is in the top-K results
6. Calculate Top-1, Top-5, Top-10 accuracy

Success means: Given an augmented view, the model correctly identifies
its original tooth scan as the most similar match.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# --- 1. SETUP PATHS ---
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent

TRAIN_DIR = project_root / "train"
DATA_DIR = project_root / "normalization"

sys.path.append(str(TRAIN_DIR))

# --- 2. IMPORT MODEL ---
try:
    from train_oasis import SimCLREncoder
except ImportError as e:
    print(f"Error importing model: {e}")
    print(f"Make sure train_oasis.py exists in {TRAIN_DIR}")
    sys.exit(1)

# --- 3. CONFIGURATION ---
# Model paths
BASE_MODEL_PATH = TRAIN_DIR / "oasis_simclr_edgeconv.pth"
FINETUNE_DIR = project_root / "fine-tuning"
MODEL_HISTORY_DIR = FINETUNE_DIR / "model_history"
LATEST_MODEL_INFO = MODEL_HISTORY_DIR / "latest.txt"

# Data paths
ORIGINAL_DATASET_PATH = DATA_DIR / "teeth3ds_dataset.npy"
VALIDATION_VIEWS_PATH = current_folder / "validation_views.npy"
VALIDATION_INDICES_PATH = current_folder / "validation_views_indices.npy"

# Evaluation settings
BATCH_SIZE = 32
TOP_K_VALUES = [1, 3, 5, 10]  # Calculate accuracy for these K values


def get_latest_finetuned_model():
    """Get the path to the latest fine-tuned model, or None if not exists."""
    if LATEST_MODEL_INFO.exists():
        with open(LATEST_MODEL_INFO, 'r') as f:
            latest_path = Path(f.read().strip())
            if latest_path.exists():
                return latest_path
    return None


def load_model(model_path, device):
    """Load the trained SimCLR encoder model."""
    print(f"Loading model from: {model_path}")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return None
    
    model = SimCLREncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def generate_embeddings(model, data, device, desc="Generating embeddings"):
    """
    Generate embeddings for a dataset using the model's backbone.
    
    Args:
        model: The SimCLR encoder model
        data: numpy array of shape (N, num_points, 3)
        device: torch device
        desc: Description for progress bar
    
    Returns:
        numpy array of embeddings, shape (N, embedding_dim)
    """
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data), BATCH_SIZE), desc=desc):
            batch = data[i:i + BATCH_SIZE]
            batch_tensor = torch.tensor(batch, dtype=torch.float32)
            # Shape: [Batch, num_points, 3] -> [Batch, 3, num_points]
            batch_tensor = batch_tensor.transpose(2, 1).to(device)
            
            # Get backbone representation (not projection head)
            representation, _ = model(batch_tensor)
            embeddings.append(representation.cpu().numpy())
    
    return np.vstack(embeddings)


def compute_similarity_matrix(query_embeddings, gallery_embeddings):
    """
    Compute cosine similarity between query and gallery embeddings.
    
    Args:
        query_embeddings: (N_query, dim) numpy array
        gallery_embeddings: (N_gallery, dim) numpy array
    
    Returns:
        similarity_matrix: (N_query, N_gallery) numpy array
    """
    # Normalize embeddings
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    similarity_matrix = np.dot(query_norm, gallery_norm.T)
    
    return similarity_matrix


def evaluate_retrieval(similarity_matrix, query_indices, top_k_values):
    """
    Evaluate retrieval accuracy.
    
    For each query (augmented view), check if its original scan
    is within the top-K most similar gallery items.
    
    Args:
        similarity_matrix: (N_query, N_gallery) similarity scores
        query_indices: (N_query,) original indices for each query
        top_k_values: list of K values to evaluate
    
    Returns:
        dict: {k: accuracy} for each k in top_k_values
    """
    num_queries = len(query_indices)
    results = {k: 0 for k in top_k_values}
    
    # For each query, get the indices sorted by similarity (descending)
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    for i, true_idx in enumerate(query_indices):
        # Get the rank of the true match
        retrieved_indices = sorted_indices[i]
        
        for k in top_k_values:
            if true_idx in retrieved_indices[:k]:
                results[k] += 1
    
    # Convert to accuracy
    accuracies = {k: count / num_queries * 100 for k, count in results.items()}
    
    return accuracies, sorted_indices


def compute_mean_reciprocal_rank(similarity_matrix, query_indices):
    """
    Compute Mean Reciprocal Rank (MRR).
    
    MRR = (1/N) * sum(1/rank_i) where rank_i is the rank of the correct answer.
    """
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    reciprocal_ranks = []
    for i, true_idx in enumerate(query_indices):
        # Find the rank of the true index (1-indexed)
        rank = np.where(sorted_indices[i] == true_idx)[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)
    
    return np.mean(reciprocal_ranks)


def print_detailed_results(accuracies, mrr, num_queries, num_gallery):
    """Print formatted validation results."""
    print("\n" + "=" * 60)
    print("              VALIDATION RESULTS")
    print("=" * 60)
    print(f"\n📊 Dataset Statistics:")
    print(f"   • Query samples (augmented views): {num_queries}")
    print(f"   • Gallery samples (original scans): {num_gallery}")
    
    print(f"\n🎯 Retrieval Accuracy:")
    for k, acc in sorted(accuracies.items()):
        bar_length = int(acc / 2)  # Scale to 50 chars max
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"   • Top-{k:2d}: {bar} {acc:6.2f}%")
    
    print(f"\n📈 Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print("=" * 60)


def analyze_failures(similarity_matrix, query_indices, sorted_indices, num_failures=5):
    """
    Analyze failure cases where top-1 retrieval failed.
    """
    print(f"\n🔍 Analyzing Top-{num_failures} Failure Cases:")
    print("-" * 40)
    
    failures = []
    for i, true_idx in enumerate(query_indices):
        top1_idx = sorted_indices[i, 0]
        if top1_idx != true_idx:
            # Find the actual rank of the true match
            rank = np.where(sorted_indices[i] == true_idx)[0][0] + 1
            failures.append({
                'query_idx': i,
                'true_idx': true_idx,
                'predicted_idx': top1_idx,
                'true_rank': rank,
                'top1_similarity': similarity_matrix[i, top1_idx],
                'true_similarity': similarity_matrix[i, true_idx]
            })
    
    # Sort by worst rank
    failures.sort(key=lambda x: x['true_rank'], reverse=True)
    
    for j, fail in enumerate(failures[:num_failures]):
        print(f"\n   Failure #{j+1}:")
        print(f"     Query Index: {fail['query_idx']}")
        print(f"     Expected: {fail['true_idx']}, Got: {fail['predicted_idx']}")
        print(f"     True Match Rank: {fail['true_rank']}")
        print(f"     Top-1 Similarity: {fail['top1_similarity']:.4f}")
        print(f"     True Similarity: {fail['true_similarity']:.4f}")
    
    print(f"\n   Total failures (Top-1): {len(failures)} / {len(query_indices)}")
    return failures


def validate(model_path=None, use_finetuned=False):
    """
    Main validation function.
    
    Args:
        model_path: Path to the model (optional, uses default if None)
        use_finetuned: If True, use the latest fine-tuned model instead of base model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running validation on device: {device}")
    
    # --- 1. Determine which model to use ---
    if model_path is not None:
        model_file = Path(model_path)
        print(f"Using specified model: {model_file}")
    elif use_finetuned:
        model_file = get_latest_finetuned_model()
        if model_file:
            print(f"Using LATEST fine-tuned model: {model_file.name}")
        else:
            print("No fine-tuned model found, falling back to base model")
            model_file = BASE_MODEL_PATH
    else:
        model_file = BASE_MODEL_PATH
        print("Using base model")
    
    # --- 2. Load Model ---
    model = load_model(model_file, device)
    if model is None:
        return
    
    # --- 3. Load Data ---
    print("\nLoading datasets...")
    
    if not ORIGINAL_DATASET_PATH.exists():
        print(f"Error: Original dataset not found at {ORIGINAL_DATASET_PATH}")
        return
    
    if not VALIDATION_VIEWS_PATH.exists():
        print(f"Error: Validation views not found at {VALIDATION_VIEWS_PATH}")
        print("Please run training first to generate validation views.")
        return
    
    original_data = np.load(ORIGINAL_DATASET_PATH)
    validation_views = np.load(VALIDATION_VIEWS_PATH)
    
    # Load indices mapping (which original scan each validation view belongs to)
    if VALIDATION_INDICES_PATH.exists():
        validation_indices = np.load(VALIDATION_INDICES_PATH)
    else:
        # If indices file doesn't exist, assume 1:1 mapping
        print("Warning: Indices file not found, assuming 1:1 mapping")
        validation_indices = np.arange(len(validation_views))
    
    print(f"   • Original scans: {len(original_data)}")
    print(f"   • Validation views: {len(validation_views)}")
    
    # --- 4. Generate Embeddings ---
    print("\nGenerating embeddings...")
    gallery_embeddings = generate_embeddings(
        model, original_data, device, desc="Gallery (original scans)"
    )
    query_embeddings = generate_embeddings(
        model, validation_views, device, desc="Queries (augmented views)"
    )
    
    print(f"   • Gallery embedding shape: {gallery_embeddings.shape}")
    print(f"   • Query embedding shape: {query_embeddings.shape}")
    
    # --- 5. Compute Similarity ---
    print("\nComputing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(query_embeddings, gallery_embeddings)
    
    # --- 6. Evaluate ---
    print("\nEvaluating retrieval performance...")
    accuracies, sorted_indices = evaluate_retrieval(
        similarity_matrix, validation_indices, TOP_K_VALUES
    )
    mrr = compute_mean_reciprocal_rank(similarity_matrix, validation_indices)
    
    # --- 7. Print Results ---
    print_detailed_results(
        accuracies, mrr, 
        num_queries=len(validation_views),
        num_gallery=len(original_data)
    )
    
    # --- 8. Analyze Failures ---
    analyze_failures(similarity_matrix, validation_indices, sorted_indices)
    
    # --- 9. Save Results ---
    results = {
        'model_path': str(model_file),
        'num_queries': len(validation_views),
        'num_gallery': len(original_data),
        'accuracies': accuracies,
        'mrr': mrr
    }
    
    results_path = current_folder / "validation_results.npy"
    np.save(results_path, results, allow_pickle=True)
    print(f"\nResults saved to: {results_path}")
    
    return results


def compare_models(model1_path=None, model2_path=None):
    """
    Compare two models' performance.
    
    Args:
        model1_path: Path to first model (default: base model)
        model2_path: Path to second model (default: latest fine-tuned)
    """
    # Resolve model 1
    if model1_path is None:
        model1 = BASE_MODEL_PATH
        model1_name = "BASE"
    else:
        model1 = Path(model1_path)
        if not model1.exists():
            # Try model_history directory
            possible = MODEL_HISTORY_DIR / model1_path
            if possible.exists():
                model1 = possible
            else:
                print(f"Error: Model not found: {model1_path}")
                return
        model1_name = model1.name
    
    # Resolve model 2
    if model2_path is None:
        model2 = get_latest_finetuned_model()
        if model2 is None:
            print("No fine-tuned model found. Cannot compare.")
            return
        model2_name = f"LATEST ({model2.name})"
    else:
        model2 = Path(model2_path)
        if not model2.exists():
            # Try model_history directory
            possible = MODEL_HISTORY_DIR / model2_path
            if possible.exists():
                model2 = possible
            else:
                print(f"Error: Model not found: {model2_path}")
                return
        model2_name = model2.name
    
    print("\n" + "=" * 60)
    print("         COMPARING TWO MODELS")
    print("=" * 60)
    
    print(f"\n--- MODEL 1: {model1_name} ---")
    results1 = validate(model_path=model1)
    
    print(f"\n--- MODEL 2: {model2_name} ---")
    results2 = validate(model_path=model2)
    
    if results1 and results2:
        print("\n" + "=" * 60)
        print("              COMPARISON SUMMARY")
        print("=" * 60)
        print(f"\n{'Metric':<15} {'Model 1':<15} {'Model 2':<15} {'Diff':<10}")
        print("-" * 55)
        
        for k in TOP_K_VALUES:
            acc1 = results1['accuracies'][k]
            acc2 = results2['accuracies'][k]
            diff = acc2 - acc1
            sign = "+" if diff > 0 else ""
            print(f"Top-{k:<10} {acc1:>12.2f}% {acc2:>12.2f}% {sign}{diff:>8.2f}%")
        
        mrr1 = results1['mrr']
        mrr2 = results2['mrr']
        diff = mrr2 - mrr1
        sign = "+" if diff > 0 else ""
        print(f"{'MRR':<15} {mrr1:>12.4f} {mrr2:>12.4f} {sign}{diff:>8.4f}")
        
        print("\n" + "-" * 55)
        print(f"Model 1: {model1}")
        print(f"Model 2: {model2}")
    else:
        print(f"\nComparison failed - one or both validations returned no results.")


def list_available_models():
    """List all available models (base + all fine-tuned versions)."""
    print("\n" + "=" * 60)
    print("  AVAILABLE MODELS")
    print("=" * 60)
    
    print(f"\n📦 Base Model:")
    print(f"   {BASE_MODEL_PATH}")
    
    if MODEL_HISTORY_DIR.exists():
        models = sorted(MODEL_HISTORY_DIR.glob("v*_*.pth"))
        if models:
            latest = get_latest_finetuned_model()
            print(f"\n📦 Fine-tuned Models ({len(models)} versions):")
            for model_path in models:
                is_latest = " ← LATEST" if model_path == latest else ""
                print(f"   {model_path.name}{is_latest}")
    else:
        print("\n📦 Fine-tuned Models: None")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate OASIS teeth similarity model")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to model file (optional)")
    parser.add_argument("--finetuned", action="store_true",
                        help="Use latest fine-tuned model instead of base model")
    parser.add_argument("--compare", nargs='*', default=None,
                        help="Compare two models. Usage: --compare [model1] [model2]. "
                             "No args = base vs latest. One arg = base vs specified. "
                             "Two args = model1 vs model2.")
    parser.add_argument("--list", action="store_true",
                        help="List all available models")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.compare is not None:
        # Parse compare arguments
        if len(args.compare) == 0:
            # --compare with no args: base vs latest
            compare_models(model1_path=None, model2_path=None)
        elif len(args.compare) == 1:
            # --compare model1: base vs model1
            compare_models(model1_path=None, model2_path=args.compare[0])
        else:
            # --compare model1 model2
            compare_models(model1_path=args.compare[0], model2_path=args.compare[1])
    else:
        validate(model_path=args.model, use_finetuned=args.finetuned)
