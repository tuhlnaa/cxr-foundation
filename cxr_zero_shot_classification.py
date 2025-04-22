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
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Optional imports for interactive use
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    INTERACTIVE_MODE = True
except ImportError:
    INTERACTIVE_MODE = False

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
    """Authenticate with HuggingFace if no token is set."""
    from huggingface_hub.utils import HfFolder

    if HfFolder.get_token() is None:
        from huggingface_hub import notebook_login
        notebook_login()
    else:
        print("HuggingFace token already set")


def load_precomputed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download and load precomputed embeddings and labels from HuggingFace.
    
    Returns:
        Tuple containing:
        - DataFrame with image embeddings
        - DataFrame with text embeddings
        - DataFrame with labels
    """
    from huggingface_hub import hf_hub_download
    
    # Download precomputed image embeddings
    embeddings_path = hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename='embeddings.npz', 
        subfolder='precomputed_embeddings'
    )
    
    with np.load(embeddings_path) as embeddings_file:
        image_embeddings_df = pd.DataFrame(
            [(key, embeddings_file[key]) for key in embeddings_file.keys()],
            columns=['image_id', 'embeddings']
        )
    
    # Download precomputed text embeddings
    text_embeddings_path = hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename='text_embeddings.npz', 
        subfolder='precomputed_embeddings'
    )
    
    with np.load(text_embeddings_path) as text_embeddings_file:
        text_embeddings_df = pd.DataFrame(
            [(key, text_embeddings_file[key]) for key in text_embeddings_file.keys()],
            columns=['query', 'embeddings']
        )
    
    # Download labels
    labels_path = hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename='labels.csv', 
        subfolder='precomputed_embeddings'
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


def run_interactive_dashboard(
    image_embeddings_df: pd.DataFrame,
    text_embeddings_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> None:
    """
    Run interactive dashboard for zero-shot classification exploration.
    
    Args:
        image_embeddings_df: DataFrame with image embeddings
        text_embeddings_df: DataFrame with text embeddings
        labels_df: DataFrame with labels
    """
    if not INTERACTIVE_MODE:
        print("Interactive widgets not available. Install ipywidgets to use this feature.")
        return
    
    text_embeddings_queries = list(text_embeddings_df['query'])
    
    diagnosis_dropdown = widgets.Dropdown(
        options=DIAGNOSIS_COLUMNS,
        description='Diagnosis:',
        disabled=False,
    )

    text_input_pos = widgets.Combobox(
        placeholder='Type positive text...',
        options=text_embeddings_queries,
        ensure_option=True
    )

    text_input_neg = widgets.Combobox(
        placeholder='Type negative text...',
        options=text_embeddings_queries,
        ensure_option=True
    )

    clear_button_pos = widgets.Button(description="Change Positive Text")
    clear_button_neg = widgets.Button(description="Change Negative Text")
    clear_button_pos.on_click(lambda b: text_input_pos.set_trait('value', ''))
    clear_button_neg.on_click(lambda b: text_input_neg.set_trait('value', ''))

    processing = False

    def draw_auc_plot(diagnosis, pos_txt=None, neg_txt=None):
        nonlocal processing
        if pos_txt == '' or neg_txt == '' or processing:
            return
            
        processing = True
        clear_output(True)
        print('Computing, please wait')
        
        if pos_txt is None or neg_txt is None:
            pos_txt, neg_txt = DIAGNOSIS_TEXT_PROMPTS[diagnosis]
            text_input_pos.value = pos_txt
            text_input_neg.value = neg_txt

        roc_auc, fpr, tpr, _ = evaluate_diagnosis(
            diagnosis, 
            labels_df, 
            image_embeddings_df, 
            text_embeddings_df,
            pos_txt, 
            neg_txt
        )

        clear_output()
        display(diagnosis_dropdown)
        
        plot_roc_curve(fpr, tpr, roc_auc, f"ROC for {diagnosis}")
        plt.show()
        
        # Display text input widgets
        display(widgets.HBox([
                widgets.Label(value="Using positive text query"),
                text_input_pos,
                clear_button_pos
            ]))
        display(widgets.HBox([
                widgets.Label(value="Negative text query "),
                text_input_neg,
                clear_button_neg
            ]))
            
        processing = False

    def update_plot(change):
        draw_auc_plot(change.new)

    def on_text_change(change):
        if change.new:
            draw_auc_plot(diagnosis_dropdown.value, text_input_pos.value, text_input_neg.value)

    diagnosis_dropdown.observe(update_plot, names='value')
    text_input_pos.observe(on_text_change, names='value')
    text_input_neg.observe(on_text_change, names='value')
    
    display(diagnosis_dropdown)
    draw_auc_plot(diagnosis_dropdown.value)


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


def main():
    """Main function to run the zero-shot classification demo."""
    authenticate_huggingface()
    
    print("Loading precomputed data...")
    image_embeddings_df, text_embeddings_df, labels_df = load_precomputed_data()
    
    if INTERACTIVE_MODE:
        print("\nStarting interactive dashboard...")
        run_interactive_dashboard(image_embeddings_df, text_embeddings_df, labels_df)
    else:
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
        plt.savefig("roc_curves.png")
        plt.show()
        
        print(f"\nOverall mean AUC: {np.mean(list(results.values())):.3f}")


if __name__ == "__main__":
    main()