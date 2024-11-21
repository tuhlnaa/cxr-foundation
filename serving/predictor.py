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
"""CXR foundation model predictor.

Prepares model input, calls the model, and post-processes the output into the
final response.
"""
import base64
import os
from typing import Any, Mapping
from absl import logging
from google.oauth2 import credentials
import numpy as np
import pydicom
from data_processing import data_processing_lib
from serving_framework import model_runner


_INPUT_BYTES_KEY = 'input_bytes'
_GCS_KEY = 'gcs_uri'
_DICOM_KEY = 'dicomweb_uri'
_PROMPT_QUERY_KEY = 'prompt_query'
_BEARER_TOKEN_KEY = 'bearer_token'

# Renamings from model outputs to API outputs.
# This allows more consistent, user-friendly names than what is returned by the
# model.
_OUTPUT_KEY_RENAMINGS = {
    'all_contrastive_img_emb': 'contrastive_img_emb',
    'img_emb': 'general_img_emb',
}


# TODO(b/372747494): Improve error handling and client-facing messaging.
class _PredictorError(Exception):
  """Exception for known predictor errors."""

  def __init__(self, client_message: str):
    super().__init__()
    self.client_message = client_message


class Predictor:
  """A predictor for getting embeddings from the CXR Foundation model."""

  def _get_image_data(
      self, instance: dict[str, Any]
  ) -> bytes | pydicom.Dataset | None:
    """Gets the image data from a request instance, or None if none is present."""
    image_keys = instance.keys() & {_GCS_KEY, _DICOM_KEY, _INPUT_BYTES_KEY}
    if len(image_keys) > 1:
      raise _PredictorError(
          'Multiple image keys found in request instance: %s' % image_keys
      )

    if _INPUT_BYTES_KEY in instance:
      return base64.b64decode(instance[_INPUT_BYTES_KEY])
    creds = (
        credentials.Credentials(token=instance[_BEARER_TOKEN_KEY])
        if _BEARER_TOKEN_KEY in instance
        else None
    )
    if _GCS_KEY in instance:
      gcs_uri = instance[_GCS_KEY]
      _, file_ext = os.path.splitext(gcs_uri)
      if file_ext == '.dcm':
        logging.info('Retrieving dicom from GCS: %s', gcs_uri)
        return data_processing_lib.retrieve_dicom_from_gcs(gcs_uri, creds)
      else:
        logging.info('Retrieving file bytes from GCS: %s', gcs_uri)
        return data_processing_lib.retrieve_file_bytes_from_gcs(gcs_uri, creds)
    if _DICOM_KEY in instance:
      dicomweb_uri = instance[_DICOM_KEY]
      logging.info('Retrieving instance from DICOM store: %s', dicomweb_uri)
      return data_processing_lib.retrieve_instance_from_dicom_store(
          dicomweb_uri, creds
      )
    return None

  def _run_elixr_c(
      self,
      image_data: bytes | pydicom.Dataset,
      cxr_model_runner: model_runner.ModelRunner,
  ) -> np.ndarray:
    """Given image data, runs Elixr-C on it and returns the output."""
    image_example = data_processing_lib.process_xray_image_to_tf_example(
        image_data
    )
    elixr_c_input = {
        'input_example': np.array([image_example.SerializeToString()])
    }
    return cxr_model_runner.run_model(
        model_input=elixr_c_input,
        model_name='elixr_c',
        model_output_key='feature_maps_0',
    )

  def _run_tokenizer(
      self,
      text: str,
      cxr_model_runner: model_runner.ModelRunner,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Tokenizes input text and returns token IDs and padding masks."""
    outs = cxr_model_runner.run_model_multiple_output(
        model_input={'input_1': np.array([text.lower()])},
        model_name='tokenizer',
        model_output_keys={'input_word_ids', 'input_mask'},
    )
    ids = outs['input_word_ids'].astype(np.int32)
    masks = outs['input_mask'].astype(np.float32)
    paddings = 1.0 - masks
    end_token_idx = ids == 102
    ids[end_token_idx] = 0
    paddings[end_token_idx] = 1.0
    ids = np.expand_dims(ids, axis=1)
    paddings = np.expand_dims(paddings, axis=1)
    return ids, paddings

  def _handle_request_instance(
      self, instance: dict[str, Any], cxr_model_runner: model_runner.ModelRunner
  ) -> dict[str, Any]:
    """Handles a single request instance."""
    # Elixr-B is composed of three models: Elixr-C, a tokenizer, and a Qformer.
    # This method runs Elixr-C on the image data, then runs the tokenizer on the
    # prompt query, and finally runs Qformer on the image and text outputs.
    #
    # If there is no image data, then zeros are used as the image input to
    # Qformer.  If there is no prompt query, then zeros are used as the text
    # input to Qformer.

    # Build a set of keys to return.
    # This is based on the data types provided - e.g., text embeddings are
    # only returned if a prompt query is provided.
    keys_to_return = set()

    # Step 1: Use Elixr-C to get the image input to Qformer.
    image_data = self._get_image_data(instance)
    if image_data is None:
      qformer_image_input = np.zeros([1, 8, 8, 1376], dtype=np.float32).tolist()
    else:
      qformer_image_input = self._run_elixr_c(image_data, cxr_model_runner)
      keys_to_return.update({'all_contrastive_img_emb', 'img_emb'})

    # Step 2: Use tokenizer to get the text input to Qformer.
    prompt_query = instance.get(_PROMPT_QUERY_KEY, None)
    if not prompt_query:
      text_ids = np.zeros([1, 1, 128], dtype=np.int32)
      text_paddings = np.zeros([1, 1, 128], dtype=np.float32)
      logging.info('Using default text input.')
    else:
      logging.info('Running tokenizer on prompt query.')
      text_ids, text_paddings = self._run_tokenizer(
          prompt_query, cxr_model_runner
      )
      keys_to_return.add('contrastive_txt_emb')

    # Step 3: Run Qformer on the image and text inputs.
    qformer_input = {
        'image_feature': qformer_image_input,
        'ids': text_ids,
        'paddings': text_paddings,
    }
    outputs = cxr_model_runner.run_model_multiple_output(
        model_input=qformer_input,
        model_name='qformer',
        model_output_keys=keys_to_return,
    )

    # Construct final response.
    # Apply key renamings where necessary.
    return {
        _OUTPUT_KEY_RENAMINGS.get(key, key): value.tolist()
        for key, value in outputs.items()
    }

  def predict(
      self,
      request: dict[str, Any],
      cxr_model_runner: model_runner.ModelRunner,
  ) -> dict[str, Any]:
    """Runs model inference on the request instances.

    Args:
      request: The parsed request json to process.
      cxr_model_runner: The model runner object to use to call the model.

    Returns:
      The response json which will be returned to the client through the
      Vertex endpoint API.
    """
    predictions: list[Mapping[str, Any]] = []
    for instance in request['instances']:
      try:
        outputs = self._handle_request_instance(instance, cxr_model_runner)
      except _PredictorError as e:
        logging.exception('Failed to get prediction for instance: %s', e)
        response = {
            'error': {
                'description': (
                    'Failed to get prediction for instance. Reason:'
                    f' {e.client_message}'
                )
            }
        }
        predictions.append(response)
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Catch-all for any other exceptions that haven't been caught and
        # converted to _PredictorError.
        logging.exception('Failed to get prediction for instance: %s', e)
        response = {
            'error': {
                'description': 'Internal error getting prediction for instance.'
            }
        }
        predictions.append(response)
      else:
        predictions.append(outputs)

    return {'predictions': predictions}
