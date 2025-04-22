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
#     <a href="https://colab.research.google.com/github/google-health/cxr-foundation/blob/master/notebooks/quick_start_with_hugging_face.ipynb">
#       <img alt="Google Colab logo" src="https://www.tensorflow.org/images/colab_logo_32px.png" width="32px"><br> Run in Google Colab
#     </a>
#   </td>
#   <td style="text-align: center">
#     <a href="https://github.com/google-health/cxr-foundation/blob/master/notebooks/quick_start_with_hugging_face.ipynb">
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
# # Quick start with Hugging Face
# This Colab notebook provides a basic demo of using Chest X-ray (CXR) Foundation. CXR Foundation is an embeddings models that generates a machine learning representations known as embeddings, from chest X-ray images and/or chest X-ray related text. These embeddings can be used to develop custom models for CXR use-cases with less data and compute compared to traditional model development methods. Learn more about embeddings and their benefits at this [page](https://developers.google.com/health-ai-developer-foundations/cxr-foundation).

# ELIXR: Towards a general purpose X-ray artificial intelligence system through alignment of large language models and radiology vision encoders
# In this work, we present an approach, which we call Embeddings for Language/Image-aligned X-Rays, or ELIXR, that leverages a language-aligned image encoder combined or grafted onto a fixed LLM, PaLM 2, to perform a broad range of chest X-ray tasks. 
# We train this lightweight adapter architecture using images paired with corresponding free-text radiology reports from the MIMIC-CXR dataset. 
# ELIXR achieved state-of-the-art performance on zero-shot chest X-ray (CXR) classification (mean AUC of 0.850 across 13 findings), data-efficient CXR classification (mean AUCs of 0.893 and 0.898 across five findings (atelectasis, cardiomegaly, consolidation, pleural effusion, and pulmonary edema) for 1% (~2,200 images) and 10% (~22,000 images) training data), and semantic search (0.76 normalized discounted cumulative gain (NDCG) across nineteen queries, including perfect retrieval on twelve of them). 
# Compared to existing data-efficient methods including supervised contrastive learning (SupCon), ELIXR required two orders of magnitude less data to reach similar performance. 
# ELIXR also showed promise on CXR vision-language tasks, demonstrating overall accuracies of 58.7% and 62.5% on visual question answering and report quality assurance tasks, respectively. 
# These results suggest that ELIXR is a robust and versatile approach to CXR AI.

# %%
# @title Authenticate with HuggingFace, skip if you have a HF_TOKEN secret

# Authenticate user for HuggingFace if needed. Enter token below if requested.
from huggingface_hub.utils import HfFolder
from huggingface_hub import notebook_login

if HfFolder.get_token() is None:
    from huggingface_hub import notebook_login
    notebook_login()

# %%
# @title Helper Functions to prepare inputs: text & image TF Example
!pip install tensorflow-text==2.17 pypng 2>&1 1>/dev/null
import io
import png
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub
import numpy as np

# Helper function for tokenizing text input
def bert_tokenize(text):
    """Tokenizes input text and returns token IDs and padding masks."""
    preprocessor = tf_hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    out = preprocessor(tf.constant([text.lower()]))
    ids = out['input_word_ids'].numpy().astype(np.int32)
    masks = out['input_mask'].numpy().astype(np.float32)
    paddings = 1.0 - masks
    end_token_idx = ids == 102
    ids[end_token_idx] = 0
    paddings[end_token_idx] = 1.0
    ids = np.expand_dims(ids, axis=1)
    paddings = np.expand_dims(paddings, axis=1)
    assert ids.shape == (1, 1, 128)
    assert paddings.shape == (1, 1, 128)
    return ids, paddings

# Helper function for processing image data
def png_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
    """Creates a tf.train.Example from a NumPy array."""
    # Convert the image to float32 and shift the minimum value to zero
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        # For uint8 images, no rescaling is needed
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        # For other data types, scale image to use the full 16-bit range
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val  # Scale to 16-bit range
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    # Ensure the array is 2-D (grayscale image)
    if pixel_array.ndim != 2:
        raise ValueError(f'Array must be 2-D. Actual dimensions: {pixel_array.ndim}')

    # Encode the array as a PNG image
    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    # Create a tf.train.Example and assign the features
    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(png_bytes)
    features['image/format'].bytes_list.value.append(b'png')

    return example

# %% [markdown]
# # Compute Embeddings

# %%
# @title Fetch Sample Image
from PIL import Image
from IPython.display import Image as IPImage, display
# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
!wget -nc -q https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png
display(IPImage(filename="Chest_Xray_PA_3-8-2010.png", height=300))
img = Image.open("Chest_Xray_PA_3-8-2010.png").convert('L')  # Convert to grayscale

# %%
# @title Invoke Model with Image
import numpy as np
import matplotlib.pyplot as plt

# Download the model repository files
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/cxr-foundation",local_dir='/content/hf',
                  allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'])

# Step 1 - ELIXR C (image to elixr C embeddings)
serialized_img_tf_example = png_to_tfexample(np.array(img)).SerializeToString()

if 'elixrc_model' not in locals():
  elixrc_model = tf.saved_model.load('/content/hf/elixr-c-v2-pooled')
  elixrc_infer = elixrc_model.signatures['serving_default']

elixrc_output = elixrc_infer(input_example=tf.constant([serialized_img_tf_example]))
elixrc_embedding = elixrc_output['feature_maps_0'].numpy()

print("ELIXR-C - interim embedding shape: ", elixrc_embedding.shape)

# Step 2 - Invoke QFormer with Elixr-C embeddings
# Initialize text inputs with zeros
qformer_input = {
    'image_feature': elixrc_embedding.tolist(),
    'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
    'paddings':np.zeros((1, 1, 128), dtype=np.float32).tolist(),
}

if 'qformer_model' not in locals():
  qformer_model = tf.saved_model.load("/content/hf/pax-elixr-b-text")

qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
elixrb_embeddings = qformer_output['all_contrastive_img_emb']

print("ELIXR-B - embedding shape: ", elixrb_embeddings.shape)

# Plot output
plt.imshow(elixrb_embeddings[0], cmap='gray')
plt.colorbar()  # Show a colorbar to understand the value distribution
plt.title('Visualization of ELIXR-B embedding output')
plt.show()


# %%
# @title Input Text Query
TEXT_QUERY = "Airspace opacity" # @param {type:"string"}

# %%
# @title Invoke Model with Text
import numpy as np

# Download the model repository files
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/cxr-foundation",local_dir='/content/hf',
                  allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'])

# Run QFormer with text only.
# Initialize image input with zeros
tokens, paddings = bert_tokenize(TEXT_QUERY)
qformer_input = {
    'image_feature': np.zeros([1, 8, 8, 1376], dtype=np.float32).tolist(),
    'ids': tokens.tolist(),
    'paddings': paddings.tolist(),
}

if 'qformer_model' not in locals():
  qformer_model = tf.saved_model.load("/content/hf/pax-elixr-b-text")

qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
text_embeddings = qformer_output['contrastive_txt_emb']

print("Text Embedding shape: ", text_embeddings.shape)
print("First 5 tokens: ", text_embeddings[0][0:5])


# %% [markdown]
# # Next steps
# 
# Explore the other [notebooks](https://github.com/google-health/cxr-foundation/blob/master/notebooks) to learn what else you can do with the model.


