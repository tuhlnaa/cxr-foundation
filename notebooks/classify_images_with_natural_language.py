# %% [markdown]
# ~~~
# Copyright 2024 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ~~~
# <table><tbody><tr>
#   <td style="text-align: center">
#     <a href="https://colab.research.google.com/github/google-health/cxr-foundation/blob/master/notebooks/classify_images_with_natural_language.ipynb">
#       <img alt="Google Colab logo" src="https://www.tensorflow.org/images/colab_logo_32px.png" width="32px"><br> Run in Google Colab
#     </a>
#   </td>
#   <td style="text-align: center">
#     <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2Fgoogle-health%2Fcxr-foundation%2Fmaster%2Fnotebooks%2Fclassify_images_with_natural_language.ipynb">
#       <img alt="Google Cloud Colab Enterprise logo" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" width="32px"><br> Run in Colab Enterprise
#     </a>
#   </td>
#   <td style="text-align: center">
#     <a href="https://github.com/google-health/cxr-foundation/blob/master/notebooks/classify_images_with_natural_language.ipynb">
#       <img alt="GitHub logo" src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" width="32px"><br> View on GitHub
#     </a>
#   </td>
#   <td style="text-align: center">
#     <a href="https://huggingface.co/google/cxr-foundation">
#       <img alt="HuggingFace logo" src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="32px"><br> View on HuggingFace
#     </a>
#   </td>
# </tr></tbody></table>
# 

# %% [markdown]
# # CXR Foundation Zero-shot Classification Demo
# 
# This notebook demonstrates how to use embeddings from the CXR Foundation model to perform zero-shot classification of chest X-ray images. The notebook covers the following:
# 
# 
# - Downloading 2737 precomputed embeddings and labels for a subset of the open-access NIH Chest X-ray14 (CXR-14) dataset.
# - Performing text-based zero-shot classification of diseases using these embeddings.
# - Evaluating the classification performance using AUC by comparing to the [CXR-14 labels](https://pmc.ncbi.nlm.nih.gov/articles/PMC10607847/).
# - Exploring the impact of different text prompts on the classification results.
# 
# The embeddings used in this demo are the *all_contrastive_img_emb* embeddings, which are text-aligned image embeddings from the Q-former output in [ELIXR-B](https://arxiv.org/abs/2308.01317). These embeddings have been precomputed to streamline the demonstration and eliminate the need for lengthy downloads.
# 
# **Note:** The CXR-14 labels used in this demo were generated through text mining and may have limitations. For more information on generating embeddings using the CXR Foundation model, refer to the [HuggingFace model card](https://huggingface.co/google/cxr-foundation), or using this [quickstart colab](https://colab.research.google.com/github/google-health/cxr-foundation/blob/master/notebooks/quick_start_with_hugging_face.ipynb).
# 

# ELIXR: Towards a general purpose X-ray artificial intelligence system through alignment of large language models and radiology vision encoders
# In this work, we present an approach, which we call Embeddings for Language/Image-aligned X-Rays, or ELIXR, that leverages a language-aligned image encoder combined or grafted onto a fixed LLM, PaLM 2, to perform a broad range of chest X-ray tasks. 
# We train this lightweight adapter architecture using images paired with corresponding free-text radiology reports from the MIMIC-CXR dataset. 
# ELIXR achieved state-of-the-art performance on zero-shot chest X-ray (CXR) classification (mean AUC of 0.850 across 13 findings), data-efficient CXR classification (mean AUCs of 0.893 and 0.898 across five findings (atelectasis, cardiomegaly, consolidation, pleural effusion, and pulmonary edema) for 1% (~2,200 images) and 10% (~22,000 images) training data), and semantic search (0.76 normalized discounted cumulative gain (NDCG) across nineteen queries, including perfect retrieval on twelve of them). 
# Compared to existing data-efficient methods including supervised contrastive learning (SupCon), ELIXR required two orders of magnitude less data to reach similar performance. 
# ELIXR also showed promise on CXR vision-language tasks, demonstrating overall accuracies of 58.7% and 62.5% on visual question answering and report quality assurance tasks, respectively. 
# These results suggest that ELIXR is a robust and versatile approach to CXR AI.

# %% [markdown]
# # Authenticate to Access Data

# %%
# @title Authenticate with HuggingFace, skip if you have a HF_TOKEN secret

# Authenticate user for HuggingFace if needed. Enter token below if requested.
from huggingface_hub.utils import HfFolder

if HfFolder.get_token() is None:
    from huggingface_hub import notebook_login
    notebook_login()
else:
    print("Token already set")

# %%
# @title Download precomputed embeddings and labels from HuggingFace

import pandas as pd
import numpy as np

from huggingface_hub import hf_hub_download
HF_REPO_ID = "google/cxr-foundation"

# Download precomputed embeddings.
EMBEDDINGS_NPZ_FILE_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename='embeddings.npz', subfolder='precomputed_embeddings')
embeddings_file = np.load(EMBEDDINGS_NPZ_FILE_PATH)
image_embeddings_df = pd.DataFrame(
    [(key, embeddings_file[key]) for key in embeddings_file.keys()],
    columns=['image_id', 'embeddings']
)
embeddings_file.close()

# Download precomputed text embeddings.
TEXT_EMBEDDINGS_NPZ_FILE_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename='text_embeddings.npz', subfolder='precomputed_embeddings')

# Download the labels file to annotate the outputs.
LABEL_FILE_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename='labels.csv', subfolder='precomputed_embeddings')

# Read text embeddings
text_embeddings_file = np.load(TEXT_EMBEDDINGS_NPZ_FILE_PATH)
text_embeddings_queries = list(text_embeddings_file.keys())
text_embeddings_df = pd.DataFrame(
    [(key, text_embeddings_file[key]) for key in text_embeddings_file.keys()],
    columns=['query', 'embeddings']
)
text_embeddings_file.close()

# Read labels
full_labels_df = pd.read_csv(LABEL_FILE_PATH)

# %%
# @title Similarity and Zero-shot Classification Functions

import numpy as np


# Load labels
labels_df = pd.read_csv(LABEL_FILE_PATH)
diagnosis_columns = ['AIRSPACE_OPACITY', 'PNEUMOTHORAX', 'EFFUSION', 'PULMONARY_EDEMA']

def softmax(x):
    """Calculates the softmax of a list of numbers."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(axis=0)

def compute_image_text_similarity(image_emb, txt_emb):
  image_emb = np.reshape(image_emb, (32, 128))
  similarities = []
  for i in range(32):
    # cosine similarity
    similarity = np.dot(image_emb[i], txt_emb)/(np.linalg.norm(image_emb[i]) * np.linalg.norm(txt_emb))
    similarities.append(similarity)
  np_sm_similarities = np.array((similarities))
  return np.max(np_sm_similarities)

def zero_shot(image_emb, pos_txt_emb,neg_txt_emb):
  pos_cosine = compute_image_text_similarity(image_emb, pos_txt_emb)
  neg_cosine = compute_image_text_similarity(image_emb, neg_txt_emb)
  return pos_cosine - neg_cosine

def get_text_embeddings_for_diagnosis(diagnosis):
  """
  This function takes a diagnosis as input and outputs the positive and negative text queries.
  """
  column_to_pos_neg = {
      'AIRSPACE_OPACITY': ('Airspace Opacity', 'no evidence of airspace disease'),
      'PNEUMOTHORAX': ('small pneumothorax', 'no pneumothorax'),
      'EFFUSION': ('large pleural effusion', 'no pleural effusion'),
      'PULMONARY_EDEMA': ('moderate pulmonary edema', 'no pulmonary edema'),
  }

  pos_txt, neg_txt = column_to_pos_neg[diagnosis]


  return pos_txt, neg_txt

def compute_similarity_scores(eval_data_df, pos_txt, neg_txt):
  pos_txt_emb = text_embeddings_df.set_index('query').loc[pos_txt, 'embeddings']
  neg_txt_emb = text_embeddings_df.set_index('query').loc[neg_txt, 'embeddings']

  # Iterate over each image_id in eval_data_df
  for index, row in eval_data_df.iterrows():
    image_id = row['image_id']
    # Get the embedding for the current image_id from image_embeddings_df
    image_embedding = image_embeddings_df[image_embeddings_df['image_id'] == image_id]['embeddings'].iloc[0]
    # Compute the similarity using the zero_shot function
    similarity_score = zero_shot(image_embedding, pos_txt_emb, neg_txt_emb)
    # Store the similarity score in a new column named 'score'
    eval_data_df.loc[index, 'score'] = similarity_score


# %%
# @title Evaluate and graph AUC

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

diagnosis_dropdown = widgets.Dropdown(
    options=diagnosis_columns,
    description='Diagnosis:',
    disabled=False,
)

text_input_pos = widgets.Combobox(
    placeholder='Type positive text...',
    options=text_embeddings_queries,
    ensure_option=True  # Ensures that the typed value is in the options
)

text_input_neg = widgets.Combobox(
    placeholder='Type negative text...',
    options=text_embeddings_queries,
    ensure_option=True  # Ensures that the typed value is in the options
)

clear_button_pos = widgets.Button(description="Change Positive Text")
clear_button_neg = widgets.Button(description="Change Negative Text")
clear_button_pos.on_click(lambda b: text_input_pos.set_trait('value', ''))
clear_button_neg.on_click(lambda b: text_input_neg.set_trait('value', ''))

processing = False

def draw_auc_plot(column, pos_txt = None, neg_txt = None):
  if pos_txt == '' or neg_txt == '':
    return
  global processing
  if processing:
    return
  processing = True
  clear_output(True)
  print('Computing, please wait')
  if pos_txt is None:
    pos_txt, neg_txt = get_text_embeddings_for_diagnosis(column)
    text_input_pos.value = pos_txt
    text_input_neg.value = neg_txt

  eval_data_df = labels_df[labels_df[column].isin([0, 1])][['image_id', column]].copy()
  eval_data_df.rename(columns={column: 'label'}, inplace=True)

  compute_similarity_scores(eval_data_df, pos_txt, neg_txt)

  clear_output()
  display(diagnosis_dropdown)

  # Assuming 'eval_data_df' is your DataFrame with 'label' and 'score' columns
  fpr, tpr, thresholds = roc_curve(eval_data_df['label'], eval_data_df['score'])
  roc_auc = auc(fpr, tpr)

  plt.figure()
  lw = 2
  plt.plot(
      fpr,
      tpr,
      color="darkorange",
      lw=lw,
      label="ROC curve (area = %0.2f)" % roc_auc,
  )
  plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title(f"ROC for {column}")
  plt.legend(loc="lower right")
  plt.show()
  # Create a horizontal box to display image and score
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
display(diagnosis_dropdown)
draw_auc_plot(diagnosis_dropdown.value)

text_input_pos.observe(on_text_change, names='value')
text_input_neg.observe(on_text_change, names='value')

# %% [markdown]
# # Next steps
# 
# Explore the other [notebooks](https://github.com/google-health/cxr-foundation/blob/master/notebooks) to learn what else you can do with the model.


