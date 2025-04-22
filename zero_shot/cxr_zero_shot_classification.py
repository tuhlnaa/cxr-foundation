"""
CXR Foundation Zero-shot Classification Module

This module demonstrates how to use embeddings from the CXR Foundation model to perform 
zero-shot classification of chest X-ray images. It includes functionality for:

- Loading precomputed embeddings and labels from a subset of the NIH Chest X-ray14 dataset
- Performing text-based zero-shot classification of diseases using these embeddings
- Evaluating classification performance using AUC metrics
- Visualizing results with ROC curves

The embeddings used are text-aligned image embeddings from the Q-former output in ELIXR-B.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from huggingface_hub.utils import HfFolder
from huggingface_hub import login
from huggingface_hub import hf_hub_download

# Constants
HF_REPO_ID = "google/cxr-foundation"
DIAGNOSIS_COLUMNS = [
    'AIRSPACE_OPACITY', 
    'PNEUMOTHORAX', 
    'EFFUSION', 
    'PULMONARY_EDEMA'
]

# Mapping of diagnosis labels to positive and negative text prompts
DIAGNOSIS_TEXT_PROMPTS = {
    'AIRSPACE_OPACITY': ('Airspace Opacity', 'no evidence of airspace disease'),
    'PNEUMOTHORAX': ('small pneumothorax', 'no pneumothorax'),
    'EFFUSION': ('large pleural effusion', 'no pleural effusion'),
    'PULMONARY_EDEMA': ('moderate pulmonary edema', 'no pulmonary edema'),
}


def authenticate_huggingface() -> None:
    """Authenticate with HuggingFace if no token is available."""
    if HfFolder.get_token() is None:
        print("Please visit: https://huggingface.co/settings/tokens")
        login()
        print("Authentication completed")
    else:
        print("HuggingFace Token already set")


def load_precomputed_data(data_dir: str = "./dataset") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download and load precomputed embeddings and labels from HuggingFace.

    Args:
        input_dir: Directory where downloaded data will be stored

    Returns:
        Tuple containing:
        - DataFrame with image embeddings
        - DataFrame with text embeddings
        - DataFrame with labels
    """
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download precomputed image embeddings
    embeddings_path = hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename='embeddings.npz', 
        subfolder='precomputed_embeddings',
        local_dir=data_dir
    )
    
    # Download precomputed text embeddings
    text_embeddings_path = hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename='text_embeddings.npz', 
        subfolder='precomputed_embeddings',
        local_dir=data_dir
    )
    
    # Download labels
    labels_path = hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename='labels.csv', 
        subfolder='precomputed_embeddings',
        local_dir=data_dir
    )
    
    with np.load(embeddings_path) as embeddings_file:
        image_embeddings_df = pd.DataFrame(
            [(key, embeddings_file[key]) for key in embeddings_file.keys()],
            columns=['image_id', 'embeddings']
        )

    with np.load(text_embeddings_path) as text_embeddings_file:
        text_embeddings_df = pd.DataFrame(
            [(key, text_embeddings_file[key]) for key in text_embeddings_file.keys()],
            columns=['query', 'embeddings']
        )

    labels_df = pd.read_csv(labels_path)
    
    return image_embeddings_df, text_embeddings_df, labels_df


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between the vectors
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def compute_image_text_similarity(image_emb: np.ndarray, text_emb: np.ndarray) -> float:
    """
    Compute similarity between image embedding and text embedding.
    
    Args:
        image_emb: Image embedding array
        text_emb: Text embedding array
        
    Returns:
        Maximum similarity score
    """
    image_emb = np.reshape(image_emb, (32, 128))
    similarities = [
        cosine_similarity(image_emb[i], text_emb) 
        for i in range(32)
    ]
    return np.max(similarities)


def zero_shot_classification(
        image_emb: np.ndarray, 
        pos_text_emb: np.ndarray, 
        neg_text_emb: np.ndarray
    ) -> float:
    """
    Perform zero-shot classification by computing difference between
    positive and negative text embedding similarities.
    
    Args:
        image_emb: Image embedding
        pos_text_emb: Positive text embedding
        neg_text_emb: Negative text embedding
        
    Returns:
        Classification score
    """
    pos_cosine = compute_image_text_similarity(image_emb, pos_text_emb)
    neg_cosine = compute_image_text_similarity(image_emb, neg_text_emb)
    return pos_cosine - neg_cosine


def get_text_embeddings_for_diagnosis(
    diagnosis: str,
    text_embeddings_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get positive and negative text embeddings for a specific diagnosis.
    
    Args:
        diagnosis: Diagnosis label
        text_embeddings_df: DataFrame containing text embeddings
        
    Returns:
        Tuple of (positive text embedding, negative text embedding)
    """
    pos_txt, neg_txt = DIAGNOSIS_TEXT_PROMPTS[diagnosis]
    
    pos_txt_emb = text_embeddings_df.set_index('query').loc[pos_txt, 'embeddings']
    neg_txt_emb = text_embeddings_df.set_index('query').loc[neg_txt, 'embeddings']
    
    return pos_txt_emb, neg_txt_emb


def compute_similarity_scores(
        eval_data_df: pd.DataFrame,
        image_embeddings_df: pd.DataFrame,
        pos_txt_emb: np.ndarray,
        neg_txt_emb: np.ndarray
    ) -> pd.DataFrame:
    """
    Compute similarity scores for each image in the evaluation dataset.
    
    Args:
        eval_data_df: DataFrame with image IDs and labels
        image_embeddings_df: DataFrame with image embeddings
        pos_txt_emb: Positive text embedding
        neg_txt_emb: Negative text embedding
        
    Returns:
        DataFrame with added similarity scores
    """
    result_df = eval_data_df.copy()
    
    for index, row in result_df.iterrows():
        image_id = row['image_id']
        image_embedding = image_embeddings_df[
            image_embeddings_df['image_id'] == image_id
        ]['embeddings'].iloc[0]
        
        score = zero_shot_classification(image_embedding, pos_txt_emb, neg_txt_emb)
        result_df.loc[index, 'score'] = score
        
    return result_df


def evaluate_diagnosis(
        diagnosis: str,
        labels_df: pd.DataFrame,
        image_embeddings_df: pd.DataFrame,
        text_embeddings_df: pd.DataFrame,
        pos_txt: Optional[str] = None,
        neg_txt: Optional[str] = None
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate classification performance for a specific diagnosis.
    
    Args:
        diagnosis: Diagnosis label to evaluate
        labels_df: DataFrame with ground truth labels
        image_embeddings_df: DataFrame with image embeddings
        text_embeddings_df: DataFrame with text embeddings
        pos_txt: Optional custom positive text prompt
        neg_txt: Optional custom negative text prompt
        
    Returns:
        Tuple containing:
        - AUC score
        - False positive rates
        - True positive rates
        - Thresholds
    """
    # Get appropriate text prompts
    if pos_txt is None or neg_txt is None:
        pos_txt, neg_txt = DIAGNOSIS_TEXT_PROMPTS[diagnosis]
    
    # Get text embeddings
    pos_txt_emb = text_embeddings_df.set_index('query').loc[pos_txt, 'embeddings']
    neg_txt_emb = text_embeddings_df.set_index('query').loc[neg_txt, 'embeddings']
    
    # Prepare evaluation data
    eval_data_df = labels_df[labels_df[diagnosis].isin([0, 1])][['image_id', diagnosis]].copy()
    eval_data_df.rename(columns={diagnosis: 'label'}, inplace=True)
    
    # Compute similarity scores
    eval_data_df = compute_similarity_scores(
        eval_data_df, 
        image_embeddings_df, 
        pos_txt_emb, 
        neg_txt_emb
    )
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(eval_data_df['label'], eval_data_df['score'])
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, fpr, tpr, thresholds


def plot_roc_curve(
        fpr: np.ndarray, 
        tpr: np.ndarray, 
        roc_auc: float, 
        title: str,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
    """
    Plot ROC curve with AUC.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under ROC curve
        title: Plot title
        ax: Optional matplotlib Axes to plot on
        
    Returns:
        Matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
        
    ax.plot(
        fpr, tpr, color="darkorange", lw=2,
        label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    return fig


def evaluate_with_custom_prompts(
        diagnosis: str,
        labels_df: pd.DataFrame,
        image_embeddings_df: pd.DataFrame,
        text_embeddings_df: pd.DataFrame,
        pos_txt: str,
        neg_txt: str
    ) -> None:
    """
    Evaluate a diagnosis with custom positive and negative text prompts.
    
    Args:
        diagnosis: Diagnosis to evaluate
        labels_df: DataFrame with ground truth labels
        image_embeddings_df: DataFrame with image embeddings
        text_embeddings_df: DataFrame with text embeddings
        pos_txt: Custom positive text prompt
        neg_txt: Custom negative text prompt
    """
    # Check if text prompts exist in embeddings
    available_queries = set(text_embeddings_df['query'])
    if pos_txt not in available_queries:
        print(f"Error: Positive text '{pos_txt}' not found in available text embeddings.")
        print(f"Available options: {', '.join(sorted(available_queries))}")
        return
    
    if neg_txt not in available_queries:
        print(f"Error: Negative text '{neg_txt}' not found in available text embeddings.")
        print(f"Available options: {', '.join(sorted(available_queries))}")
        return
    
    print(f"Evaluating {diagnosis} with custom prompts:")
    print(f"  Positive: '{pos_txt}'")
    print(f"  Negative: '{neg_txt}'")
    
    roc_auc, fpr, tpr, _ = evaluate_diagnosis(
        diagnosis, 
        labels_df, 
        image_embeddings_df, 
        text_embeddings_df,
        pos_txt, 
        neg_txt
    )
    
    print(f"AUC: {roc_auc:.3f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr, roc_auc, f"ROC for {diagnosis}")
    plt.savefig(f"{diagnosis}_custom_roc.png")
    plt.show()


def evaluate_all_diagnoses(
        image_embeddings_df: pd.DataFrame,
        text_embeddings_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Dict[str, float]:
    """
    Evaluate all diagnoses and return AUC scores.
    
    Args:
        image_embeddings_df: DataFrame with image embeddings
        text_embeddings_df: DataFrame with text embeddings
        labels_df: DataFrame with labels
        
    Returns:
        Dictionary mapping diagnosis labels to AUC scores
    """
    results = {}
    
    for diagnosis in DIAGNOSIS_COLUMNS:
        auc_score, _, _, _ = evaluate_diagnosis(
            diagnosis, 
            labels_df, 
            image_embeddings_df, 
            text_embeddings_df
        )
        results[diagnosis] = auc_score
        print(f"{diagnosis}: AUC = {auc_score:.3f}")
    
    return results


def list_available_text_queries(text_embeddings_df: pd.DataFrame) -> None:
    """
    List all available text queries in the text embeddings.
    
    Args:
        text_embeddings_df: DataFrame with text embeddings
    """
    print("Available text queries:")
    for query in sorted(text_embeddings_df['query']):
        print(f"  - '{query}'")


def main():
    """Main function to run the zero-shot classification demo."""
    parser = argparse.ArgumentParser(description='CXR Foundation Zero-shot Classification')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'single', 'custom', 'list-queries'],
                        help='Evaluation mode: all diagnoses, single diagnosis, custom prompts, or list available queries')
    parser.add_argument('--diagnosis', type=str, choices=DIAGNOSIS_COLUMNS,
                        help='Diagnosis to evaluate (required for single and custom modes)')
    parser.add_argument('--pos-text', type=str, help='Custom positive text query')
    parser.add_argument('--neg-text', type=str, help='Custom negative text query')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save output files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Always authenticate and load data
    authenticate_huggingface()
    image_embeddings_df, text_embeddings_df, labels_df = load_precomputed_data()
    
    if args.mode == 'list-queries':
        list_available_text_queries(text_embeddings_df)
        return
    
    if args.mode == 'single':
        if args.diagnosis is None:
            parser.error("--diagnosis is required when mode is 'single'")
            
        diagnosis = args.diagnosis
        pos_txt, neg_txt = DIAGNOSIS_TEXT_PROMPTS[diagnosis]
        
        print(f"Evaluating diagnosis: {diagnosis}")
        print(f"Using default prompts:")
        print(f"  Positive: '{pos_txt}'")
        print(f"  Negative: '{neg_txt}'")
        
        auc_score, fpr, tpr, _ = evaluate_diagnosis(
            diagnosis, 
            labels_df, 
            image_embeddings_df, 
            text_embeddings_df
        )
        
        print(f"AUC: {auc_score:.3f}")
        
        # Plot and save ROC curve
        plt.figure(figsize=(8, 6))
        plot_roc_curve(fpr, tpr, auc_score, f"ROC for {diagnosis}")
        output_path = os.path.join(args.output_dir, f"{diagnosis}_roc.png")
        plt.savefig(output_path)
        print(f"ROC curve saved to {output_path}")
        plt.close()
        
    elif args.mode == 'custom':
        if args.diagnosis is None or args.pos_text is None or args.neg_text is None:
            parser.error("--diagnosis, --pos-text, and --neg-text are required when mode is 'custom'")
            
        evaluate_with_custom_prompts(
            args.diagnosis,
            labels_df,
            image_embeddings_df,
            text_embeddings_df,
            args.pos_text,
            args.neg_text
        )
        
        # Save ROC curve
        output_path = os.path.join(args.output_dir, f"{args.diagnosis}_custom_roc.png")
        plt.savefig(output_path)
        print(f"ROC curve saved to {output_path}")
        plt.close()
        
    else:  # args.mode == 'all'
        print("\nEvaluating all diagnoses...")
        results = evaluate_all_diagnoses(image_embeddings_df, text_embeddings_df, labels_df)
        
        # Plot ROC curves for all diagnoses
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, diagnosis in enumerate(DIAGNOSIS_COLUMNS):
            auc_score = results[diagnosis]
            pos_txt, neg_txt = DIAGNOSIS_TEXT_PROMPTS[diagnosis]
            
            # Get text embeddings
            pos_txt_emb = text_embeddings_df.set_index('query').loc[pos_txt, 'embeddings']
            neg_txt_emb = text_embeddings_df.set_index('query').loc[neg_txt, 'embeddings']
            
            # Prepare evaluation data
            eval_data_df = labels_df[labels_df[diagnosis].isin([0, 1])][
                ['image_id', diagnosis]
            ].copy()
            eval_data_df.rename(columns={diagnosis: 'label'}, inplace=True)
            
            # Compute similarity scores
            eval_data_df = compute_similarity_scores(
                eval_data_df, 
                image_embeddings_df, 
                pos_txt_emb, 
                neg_txt_emb
            )
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(eval_data_df['label'], eval_data_df['score'])
            
            # Plot ROC curve
            plot_roc_curve(fpr, tpr, auc_score, f"ROC for {diagnosis}", axes[i])
        
        plt.tight_layout()
        output_path = os.path.join(args.output_dir, "all_roc_curves.png")
        plt.savefig(output_path)
        print(f"All ROC curves saved to {output_path}")
        plt.close()
        
        print(f"\nOverall mean AUC: {np.mean(list(results.values())):.3f}")
        
        # Save results to CSV
        results_df = pd.DataFrame([
            {"diagnosis": diagnosis, "auc": auc}
            for diagnosis, auc in results.items()
        ])
        
        # Replace append with pd.concat
        mean_row = pd.DataFrame([{
            "diagnosis": "MEAN", 
            "auc": np.mean(list(results.values()))
        }])
        results_df = pd.concat([results_df, mean_row], ignore_index=True)
        
        csv_path = os.path.join(args.output_dir, "auc_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()

"""
# Evaluate all diagnoses (default)
python cxr_zero_shot_classification.py

# Evaluate a single diagnosis with default prompts
python cxr_zero_shot_classification.py --mode single --diagnosis PNEUMOTHORAX

# Evaluate with custom text prompts
python cxr_zero_shot_classification.py --mode custom --diagnosis PNEUMOTHORAX --pos-text "small pneumothorax" --neg-text "no pneumothorax"

# List all available text queries
python cxr_zero_shot_classification.py --mode list-queries

# Specify output directory
python cxr_zero_shot_classification.py --output-dir my_results
"""