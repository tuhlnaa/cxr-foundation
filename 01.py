"""
CXR Foundation: A module for chest X-ray image and text embedding generation.

This module provides utilities to generate embeddings from chest X-ray images
and related text using the ELIXR model. These embeddings can be used for various
downstream tasks such as classification, semantic search, and more.

For more information, visit:
https://developers.google.com/health-ai-developer-foundations/cxr-foundation
"""

import io
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, Any

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required for BERT preprocessing
from huggingface_hub import snapshot_download
from PIL import Image


class CXRFoundation:
    """Main class for working with ELIXR CXR Foundation models."""

    MODEL_REPO_ID = "google/cxr-foundation"
    ELIXR_C_PATH = "elixr-c-v2-pooled"
    ELIXR_B_TEXT_PATH = "pax-elixr-b-text"
    BERT_PREPROCESSOR_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

    def __init__(self, model_dir: Optional[str] = None):
        """Initialize the CXR Foundation models."""
        self.model_dir = model_dir or "./tmp/hf"
        self._load_models()


    def _load_models(self) -> None:
        """Download and load the ELIXR models."""
        # Download models if not already available
        snapshot_download(
            repo_id=self.MODEL_REPO_ID,
            local_dir=self.model_dir,
            allow_patterns=[f'{self.ELIXR_C_PATH}/*', f'{self.ELIXR_B_TEXT_PATH}/*']
        )

        # Load ELIXR C model
        elixrc_model_path = Path(self.model_dir) / self.ELIXR_C_PATH
        self.elixrc_model = tf.saved_model.load(str(elixrc_model_path))
        self.elixrc_infer = self.elixrc_model.signatures['serving_default']

        # Load QFormer model
        qformer_model_path = Path(self.model_dir) / self.ELIXR_B_TEXT_PATH
        self.qformer_model = tf.saved_model.load(str(qformer_model_path))
        self.qformer_infer = self.qformer_model.signatures['serving_default']

        # Load BERT preprocessor
        self.bert_preprocessor = hub.KerasLayer(self.BERT_PREPROCESSOR_URL)


    def tokenize_text(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenize input text for the ELIXR model.

        Args:
            text: Input text to tokenize.

        Returns:
            Tuple of (token_ids, padding_masks) as numpy arrays.
        """
        text = text.lower()
        out = self.bert_preprocessor(tf.constant([text]))
       
        # Extract and process token IDs
        ids = out['input_word_ids'].numpy().astype(np.int32)
        masks = out['input_mask'].numpy().astype(np.float32)
        paddings = 1.0 - masks
       
        # Replace end tokens with padding
        end_token_idx = ids == 102
        ids[end_token_idx] = 0
        paddings[end_token_idx] = 1.0
       
        # Reshape for model input
        ids = np.expand_dims(ids, axis=1)
        paddings = np.expand_dims(paddings, axis=1)
       
        return ids, paddings


    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> tf.train.Example:
        """Preprocess an image for the ELIXR model.

        Args:
            image: Input image as a file path, numpy array, or PIL Image.

        Returns:
            A TF Example containing the processed image.

        Raises:
            ValueError: If the image format is unsupported or invalid.
        """
        # Convert different input types to numpy array
        if isinstance(image, str):
            image = Image.open(image).convert('L')  # Convert to grayscale
            image_array = np.array(image)
        elif isinstance(image, Image.Image):
            image = image.convert('L')  # Ensure grayscale
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image
            if image_array.ndim > 2:
                # Convert RGB to grayscale if needed
                if image_array.ndim == 3 and image_array.shape[2] in (3, 4):
                    image_array = np.mean(image_array[:, :, :3], axis=2).astype(image_array.dtype)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Ensure image is 2D
        if image_array.ndim != 2:
            raise ValueError(f"Processed image must be 2D. Got shape: {image_array.shape}")

        return self._create_tf_example(image_array)


    def _create_tf_example(self, image_array: np.ndarray) -> tf.train.Example:
        """Create a TF Example from a NumPy array.

        Args:
            image_array: 2D numpy array representing a grayscale image.

        Returns:
            A TF Example containing the processed image.
        """
        # Convert to float32 and shift minimum to zero
        image = image_array.astype(np.float32)
        image -= image.min()

        # Determine bit depth based on input type
        if image_array.dtype == np.uint8:
            pixel_array = image.astype(np.uint8)
            mode = 'L'  # 8-bit grayscale
        else:
            # Scale to 16-bit range
            max_val = image.max()
            if max_val > 0:
                image *= 65535 / max_val
            pixel_array = image.astype(np.uint16)
            mode = 'I;16'  # 16-bit grayscale

        # Encode the array as PNG using PIL instead of png library
        output = io.BytesIO()
        pil_image = Image.fromarray(pixel_array, mode=mode)
        pil_image.save(output, format='PNG')
        png_bytes = output.getvalue()

        # Create TF Example
        example = tf.train.Example()
        features = example.features.feature
        features['image/encoded'].bytes_list.value.append(png_bytes)
        features['image/format'].bytes_list.value.append(b'png')

        return example


    def get_image_embedding(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Generate embedding for a chest X-ray image.

        Args:
            image: Input image as a file path, numpy array, or PIL Image.

        Returns:
            ELIXR-B embedding for the image.
        """
        # Preprocess image
        tf_example = self.preprocess_image(image)
        serialized_example = tf_example.SerializeToString()

        # Generate ELIXR-C embedding
        elixrc_output = self.elixrc_infer(input_example=tf.constant([serialized_example]))
        elixrc_embedding = elixrc_output['feature_maps_0'].numpy()

        # Generate final embedding using QFormer
        qformer_input = {
            'image_feature': elixrc_embedding.tolist(),
            'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
            'paddings': np.zeros((1, 1, 128), dtype=np.float32).tolist(),
        }
        qformer_output = self.qformer_infer(**qformer_input)
       
        return qformer_output['all_contrastive_img_emb'].numpy()


    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for chest X-ray related text.

        Args:
            text: Input text query.

        Returns:
            ELIXR-B embedding for the text.
        """
        # Tokenize text
        tokens, paddings = self.tokenize_text(text)
       
        # Generate text embedding using QFormer
        qformer_input = {
            'image_feature': np.zeros([1, 8, 8, 1376], dtype=np.float32).tolist(),
            'ids': tokens.tolist(),
            'paddings': paddings.tolist(),
        }
        qformer_output = self.qformer_infer(**qformer_input)
       
        return qformer_output['contrastive_txt_emb'].numpy()


    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding.
            emb2: Second embedding.

        Returns:
            Cosine similarity score.
        """
        # Flatten embeddings if needed
        if emb1.ndim > 2:
            emb1 = emb1.reshape(emb1.shape[0], -1)
        if emb2.ndim > 2:
            emb2 = emb2.reshape(emb2.shape[0], -1)
           
        # Compute cosine similarity
        norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
       
        return np.sum(emb1 * emb2) / (norm1 * norm2)


def authenticate_huggingface():
    """Authenticate with HuggingFace if not already authenticated."""
    from huggingface_hub.utils import HfFolder
    from huggingface_hub import notebook_login
   
    if HfFolder.get_token() is None:
        notebook_login()


def example_usage():
    """Demonstrate typical usage of the CXR Foundation model."""
    # Download a sample chest X-ray image
    import urllib.request
    import matplotlib.pyplot as plt
   
    # Download sample image
    sample_img_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    sample_img_path = "/tmp/sample_cxr.png"
    urllib.request.urlretrieve(sample_img_url, sample_img_path)
   
    # Initialize model
    cxr_model = CXRFoundation()
   
    # Generate image embedding
    img = Image.open(sample_img_path).convert('L')
    img_embedding = cxr_model.get_image_embedding(img)
    print(f"Image embedding shape: {img_embedding.shape}")
   
    # Generate text embedding
    text_query = "Airspace opacity"
    text_embedding = cxr_model.get_text_embedding(text_query)
    print(f"Text embedding shape: {text_embedding.shape}")
   
    # Visualize embedding
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Sample Chest X-ray")
   
    plt.subplot(1, 2, 2)
    plt.imshow(img_embedding[0], cmap='viridis')
    plt.colorbar()
    plt.title("ELIXR-B Embedding Visualization")
   
    plt.tight_layout()
    plt.savefig("Sample Chest X-ray.png", dpi=150)


if __name__ == "__main__":
    authenticate_huggingface()
    example_usage()