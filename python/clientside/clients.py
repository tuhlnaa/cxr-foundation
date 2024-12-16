#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Client classes for CXR Foundation.

This provides an interface, CxrClient, for generating CXR embeddings.
This provides two implementations, VertexCxrClient and HuggingFaceCxrClient.

VertexCxrClient issues requests to a CXR Vertex endpoint.
HuggingFaceCxrClient downloads Hugging Face models and run them locally.
"""

import abc
import base64
from collections.abc import Sequence
import dataclasses
import io
import os

import google.auth
import google.auth.transport.requests
from google.cloud import aiplatform
import huggingface_hub
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text  # pylint: disable=unused-import

from data_processing import data_processing_lib


_BERT_TF_HUB_PATH = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
_BERT_SEP_TOKEN_ID = 102

_ELIXR_C_EMBEDDING_SHAPE = (1, 8, 8, 1376)
_TEXT_EMBEDDING_SHAPE = (1, 1, 128)


@dataclasses.dataclass
class ImageEmbedding:
  """Data class for image embeddings.

  Attributes:
    general_img_emb: An embedding for the image that can be used for general
      purposes other than contrasting against text.
    contrastive_img_emb: A contrastive image embedding that can be used with a
      TextEmbedding contrastive_txt_emb.
    error: An error message, if any.  If present, no embeddings should be set.
  """

  general_img_emb: Sequence[float] | None = None
  contrastive_img_emb: Sequence[float] | None = None
  error: str | None = None


@dataclasses.dataclass
class TextEmbedding:
  """Data class for text embeddings.

  Attributes:
    contrastive_txt_emb: A contrastive text embedding that can be used with an
      ImageEmbedding contrastive_img_emb.
    error: An error message, if any.  If present, no embeddings should be set.
  """

  contrastive_txt_emb: Sequence[float] | None = None
  error: str | None = None


class CxrClient(abc.ABC):
  """Abstract base class for CXR clients."""

  @abc.abstractmethod
  def get_text_embeddings(
      self, strings: Sequence[str]
  ) -> Sequence[TextEmbedding]:
    """Returns a sequence of text embeddings corresponding to the sequence of strings given."""

  @abc.abstractmethod
  def get_image_embeddings_from_dicomweb(
      self, dicomweb_uris: Sequence[str]
  ) -> Sequence[ImageEmbedding]:
    """Returns a sequence of image embeddings corresponding to the sequence of DICOMweb uris given."""

  @abc.abstractmethod
  def get_image_embeddings_from_gcs(
      self, gcs_uris: Sequence[str]
  ) -> Sequence[ImageEmbedding]:
    """Returns a sequence of image embeddings corresponding to the sequence of GCS uris given."""

  @abc.abstractmethod
  def get_image_embeddings_from_images(
      self, images: Sequence[Image]
  ) -> Sequence[ImageEmbedding]:
    """Returns a sequence of image embeddings corresponding to the sequence of PIL Images given."""


def _get_gcloud_creds() -> google.auth.credentials.Credentials:
  """Returns a GCloud token for the current user."""
  creds, _ = google.auth.default()
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  return creds


class VertexCxrClient(CxrClient):
  """Client for a Vertex CXR endpoint."""

  def __init__(self, endpoint_name: str, project: str, location: str):
    """Creates a client for the given endpoint, specified by endpoint_name, project, and location.

    For more on creating a CXR Vertex endpoint, see
    https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/cxr-foundation

    Args:
      endpoint_name: The name of the endpoint - this is a numeric ID like "130"
      project: The id of the project containing the endpoint.
      location: The location of the endpoint.
    """
    self._endpoint = aiplatform.Endpoint(
        endpoint_name=endpoint_name,
        project=project,
        location=location,
    )

  def get_text_embeddings(
      self, strings: Sequence[str]
  ) -> Sequence[TextEmbedding]:
    """Returns a sequence of text embeddings corresponding to the sequence of strings given."""
    response = self._endpoint.predict(
        instances=[{'prompt_query': text} for text in strings]
    )
    return [TextEmbedding(**prediction) for prediction in response.predictions]

  def get_image_embeddings_from_dicomweb(
      self, dicomweb_uris: Sequence[str]
  ) -> Sequence[ImageEmbedding]:
    """Returns a sequence of image embeddings corresponding to the sequence of DICOMweb uris given."""
    creds = _get_gcloud_creds()
    response = self._endpoint.predict(
        instances=[
            {'dicomweb_uri': dicomweb_uri, 'bearer_token': creds.token}
            for dicomweb_uri in dicomweb_uris
        ]
    )
    return [ImageEmbedding(**prediction) for prediction in response.predictions]

  def get_image_embeddings_from_gcs(
      self, gcs_uris: Sequence[str]
  ) -> Sequence[ImageEmbedding]:
    """Returns a sequence of image embeddings corresponding to the sequence of GCS uris given."""
    creds = _get_gcloud_creds()
    response = self._endpoint.predict(
        instances=[
            {'gcs_uri': gcs_uri, 'bearer_token': creds.token}
            for gcs_uri in gcs_uris
        ]
    )
    return [ImageEmbedding(**prediction) for prediction in response.predictions]

  @classmethod
  def _image_to_base64_encoded_bytestring(cls, image: Image.Image) -> str:
    """Converts a PIL Image to a byte string."""
    with io.BytesIO() as output:
      image.save(output, format='PNG')
      return base64.b64encode(output.getvalue()).decode('utf-8')

  def get_image_embeddings_from_images(
      self, images: Sequence[Image]
  ) -> Sequence[ImageEmbedding]:
    """Returns a sequence of image embeddings corresponding to the sequence of base-64 encoded images given."""
    response = self._endpoint.predict(
        instances=[
            {
                'input_bytes': (
                    VertexCxrClient._image_to_base64_encoded_bytestring(image)
                ),
            }
            for image in images
        ]
    )
    return [ImageEmbedding(**prediction) for prediction in response.predictions]


class HuggingFaceCxrClient(CxrClient):
  """Client for a Hugging Face CXR endpoint."""

  def __init__(
      self,
      bert_model,
      elixr_c_model,
      qformer_model,
  ):
    """Creates a client with the given models.

    The CXR Foundation uses 3 models:
    - The BERT model tokenizes text
    - The ELIXR-C model generates image embeddings
    - The QFormer synthesizes text and image embeddings for use by an LLM.

    Args:
      bert_model: The BERT model.
      elixr_c_model: The ELIXR-C model.
      qformer_model: The QFormer model.
    """
    self._qformer_model = qformer_model
    self._elixr_c_model = elixr_c_model
    self._bert_model = bert_model

  def _bert_tokenize(self, text):
    """Tokenizes input text and returns token IDs and padding masks."""
    out = self._bert_model(tf.constant([text.lower()]))
    ids = out['input_word_ids'].numpy().astype(np.int32)
    masks = out['input_mask'].numpy().astype(np.float32)
    paddings = 1.0 - masks
    end_token_idx = ids == _BERT_SEP_TOKEN_ID
    ids[end_token_idx] = 0
    paddings[end_token_idx] = 1.0
    ids = np.expand_dims(ids, axis=1)
    paddings = np.expand_dims(paddings, axis=1)
    return ids, paddings

  def _get_text_embedding(self, text: str) -> TextEmbedding:
    ids, paddings = self._bert_tokenize(text)
    model_output = self._qformer_model.signatures['serving_default'](
        ids=tf.constant(ids),
        paddings=tf.constant(paddings),
        image_feature=np.zeros(
            _ELIXR_C_EMBEDDING_SHAPE, dtype=np.float32
        ).tolist(),
    )
    return TextEmbedding(
        model_output['contrastive_txt_emb'][0].numpy().tolist()
    )

  def get_text_embeddings(
      self, strings: Sequence[str]
  ) -> Sequence[TextEmbedding]:
    """Returns a sequence of text embeddings corresponding to the sequence of strings given."""
    return [self._get_text_embedding(text) for text in strings]

  def _get_image_embedding_from_dicom_web(
      self, dicomweb_uri: str
  ) -> ImageEmbedding:
    """Returns an ImageEmbedding for the given DICOMweb uri."""
    creds = _get_gcloud_creds()
    dicom_dataset = data_processing_lib.retrieve_instance_from_dicom_store(
        dicomweb_uri, creds
    )
    example = data_processing_lib.process_xray_image_to_tf_example(
        dicom_dataset
    )

    elixr_c_output = self._elixr_c_model.signatures['serving_default'](
        input_example=tf.constant([example.SerializeToString()])
    )
    qformer_output = self._qformer_model.signatures['serving_default'](
        image_feature=elixr_c_output['feature_maps_0'].numpy().tolist(),
        ids=np.zeros(_TEXT_EMBEDDING_SHAPE, dtype=np.int32).tolist(),
        paddings=np.zeros(_TEXT_EMBEDDING_SHAPE, dtype=np.float32).tolist(),
    )
    return ImageEmbedding(
        general_img_emb=qformer_output['img_emb'].numpy()[0].tolist(),
        contrastive_img_emb=qformer_output['all_contrastive_img_emb']
        .numpy()[0]
        .tolist(),
    )

  def get_image_embeddings_from_dicomweb(
      self, dicomweb_uris: Sequence[str]
  ) -> Sequence[ImageEmbedding]:
    """Returns a sequence of image embeddings corresponding to the sequence of DICOMweb uris given."""
    return [
        self._get_image_embedding_from_dicom_web(dicomweb_uri)
        for dicomweb_uri in dicomweb_uris
    ]

  def _get_image_example_from_gcs(self, gcs_uri: str) -> tf.train.Example:
    """Retrieves an image from GCS and converts it to a TF example.

    Args:
      gcs_uri: The GCS uri of the image.  This can be either a DICOM file or a
        PNG file.

    Returns:
      A TF example containing the image.
    """
    creds = _get_gcloud_creds()
    _, file_ext = os.path.splitext(gcs_uri)
    if file_ext == '.dcm':
      dicom_dataset = data_processing_lib.retrieve_dicom_from_gcs(
          gcs_uri, creds
      )
      return data_processing_lib.process_xray_image_to_tf_example(dicom_dataset)
    else:
      img_bytes = data_processing_lib.retrieve_file_bytes_from_gcs(
          gcs_uri, creds
      )
      return data_processing_lib.process_xray_image_to_tf_example(img_bytes)

  def _get_image_embedding_from_gcs(self, gcs_uri: str) -> ImageEmbedding:
    """Retrieves an image from GCS and converts it to an ImageEmbedding.

    Args:
      gcs_uri: The GCS uri of the image.  This can be either a DICOM file or a
        PNG file.
    """

    example = self._get_image_example_from_gcs(gcs_uri)
    elixr_c_output = self._elixr_c_model.signatures['serving_default'](
        input_example=tf.constant([example.SerializeToString()])
    )
    model_output = self._qformer_model.signatures['serving_default'](
        image_feature=elixr_c_output['feature_maps_0'].numpy().tolist(),
        ids=np.zeros(_TEXT_EMBEDDING_SHAPE, dtype=np.int32).tolist(),
        paddings=np.zeros(_TEXT_EMBEDDING_SHAPE, dtype=np.float32).tolist(),
    )
    return ImageEmbedding(
        general_img_emb=model_output['img_emb'].numpy()[0].tolist(),
        contrastive_img_emb=(
            model_output['all_contrastive_img_emb'].numpy()[0].tolist()
        ),
    )

  def get_image_embeddings_from_gcs(
      self, gcs_uris: Sequence[str]
  ) -> Sequence[ImageEmbedding]:
    """Returns a sequence of image embeddings corresponding to the sequence of GCS uris given.

    Args:
      gcs_uris: The GCS uris of the images.  This can be either a DICOM file or
        a PNG file.
    """
    return [self._get_image_embedding_from_gcs(gcs_uri) for gcs_uri in gcs_uris]

  def _get_image_embedding_from_image(self, image: Image) -> ImageEmbedding:
    """Converts a PIL Image to an ImageEmbedding."""
    with io.BytesIO() as buffer:
      image.save(buffer, format='PNG')
      image_bytes = buffer.getvalue()
    image_example = data_processing_lib.process_xray_image_to_tf_example(
        image_bytes
    )
    elixr_c_output = self._elixr_c_model.signatures['serving_default'](
        input_example=tf.constant([image_example.SerializeToString()])
    )
    model_output = self._qformer_model.signatures['serving_default'](
        image_feature=elixr_c_output['feature_maps_0'].numpy().tolist(),
        ids=np.zeros(_TEXT_EMBEDDING_SHAPE, dtype=np.int32).tolist(),
        paddings=np.zeros(_TEXT_EMBEDDING_SHAPE, dtype=np.float32).tolist(),
    )
    return ImageEmbedding(
        general_img_emb=model_output['img_emb'].numpy()[0].tolist(),
        contrastive_img_emb=(
            model_output['all_contrastive_img_emb'].numpy()[0].tolist()
        ),
    )

  def get_image_embeddings_from_images(
      self, images: Sequence[Image]
  ) -> Sequence[ImageEmbedding]:
    """Returns a sequence of image embeddings corresponding to the sequence of base-64 encoded images given."""
    return [self._get_image_embedding_from_image(image) for image in images]


def make_hugging_face_client(model_dir: str) -> HuggingFaceCxrClient:
  """Downloads the Hugging Face models and returns a HuggingFaceCxrClient."""
  huggingface_hub.snapshot_download(
      repo_id='google/cxr-foundation',
      local_dir=model_dir,
      allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'],
  )
  elixr_c_model = tf.saved_model.load(f'{model_dir}/elixr-c-v2-pooled')
  qformer_model = tf.saved_model.load(f'{model_dir}/pax-elixr-b-text')
  bert_model = tf_hub.KerasLayer(_BERT_TF_HUB_PATH)
  return HuggingFaceCxrClient(bert_model, elixr_c_model, qformer_model)
