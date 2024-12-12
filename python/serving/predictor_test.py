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

"""Tests for the CXR foundation model predictor."""

from unittest import mock

import numpy as np
import pydicom
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized
from data_processing import data_processing_lib
from serving.serving_framework import model_runner
from serving import predictor

_TEST_IMAGE_EXAMPLE = tf.train.Example(
    features=tf.train.Features(
        feature={
            "some-image-key": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[b"image_bytes"])
            )
        }
    )
)


# Wrapper for np.array for use in mocks call tests. This is necessary because
# mock.assert_called_with will use `==` natively, rather than np.array_equal.
# This enables calls like:
# mock.assert_called_with(
#   model_input={
#     "image_feature": ArrayThatIsEqualTo(
#       np.zeros([1, 8, 8, 1376], dtype=np.float32).tolist()
#     ),
#   },
# )
class ArrayThatIsEqualTo:

  def __init__(self, val: np.ndarray):
    self._val = val

  def __eq__(self, other: np.ndarray):
    return np.array_equal(self._val, other)


class PredictorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="dicom_from_gcs",
          request={
              "instances": [{
                  "gcs_uri": "fake_dicom.dcm",
              }]
          },
          mock_retrieve_dicom_from_gcs_calls=1,
          expect_tokenization=False,
          expect_elixr_c_call=True,
          expected_output_keys={"all_contrastive_img_emb", "img_emb"},
      ),
      dict(
          testcase_name="png_from_gcs",
          request={
              "instances": [{
                  "gcs_uri": "fake_dicom.png",
              }]
          },
          mock_retrieve_file_bytes_from_gcs_calls=1,
          expect_tokenization=False,
          expect_elixr_c_call=True,
          expected_output_keys={"all_contrastive_img_emb", "img_emb"},
      ),
      dict(
          testcase_name="dicom_from_dicomweb",
          request={
              "instances": [{
                  "dicomweb_uri": "fake_dicom_web_uri",
              }]
          },
          mock_retrieve_instance_from_dicom_store_calls=1,
          expect_tokenization=False,
          expect_elixr_c_call=True,
          expected_output_keys={"all_contrastive_img_emb", "img_emb"},
      ),
      dict(
          testcase_name="raw_bytes",
          request={
              "instances": [{
                  "input_bytes": "c29tZV9ieXRlcw==",
              }]
          },
          expect_tokenization=False,
          # No retrieval calls, but still expect elixr_c call.
          expect_elixr_c_call=True,
          expected_output_keys={"all_contrastive_img_emb", "img_emb"},
      ),
      dict(
          testcase_name="text_only",
          request={
              "instances": [{
                  "prompt_query": "fake_prompt_query",
              }]
          },
          expect_tokenization=True,
          expect_elixr_c_call=False,
          expected_output_keys={"contrastive_txt_emb"},
      ),
      dict(
          # Test a query with image data AND text data.
          # Note that image data since the two are processed separately,
          # it's not necessary to have "image_and_text" tests for each image
          # type.
          testcase_name="image_and_text",
          request={
              "instances": [{
                  "dicomweb_uri": "fake_dicom_web_uri",
                  "prompt_query": "fake_prompt_query",
              }]
          },
          mock_retrieve_instance_from_dicom_store_calls=1,
          expect_tokenization=True,
          expect_elixr_c_call=True,
          expected_output_keys={
              "all_contrastive_img_emb",
              "img_emb",
              "contrastive_txt_emb",
          },
      ),
  )
  @mock.patch.object(
      data_processing_lib,
      "process_xray_image_to_tf_example",
      autospec=True,
      return_value=_TEST_IMAGE_EXAMPLE,
  )
  @mock.patch.object(
      data_processing_lib,
      "retrieve_dicom_from_gcs",
      autospec=True,
      return_value=pydicom.Dataset(),
  )
  @mock.patch.object(
      data_processing_lib,
      "retrieve_file_bytes_from_gcs",
      autospec=True,
      return_value=b"file_bytes",
  )
  @mock.patch.object(
      data_processing_lib,
      "retrieve_instance_from_dicom_store",
      autospec=True,
      return_value=pydicom.Dataset(),
  )
  @mock.patch.object(model_runner, "ModelRunner", autospec=True)
  def test_predict_with_one_instance_success(
      self,
      mock_model_runner,
      mock_retrieve_instance_from_dicom_store,
      mock_retrieve_file_bytes_from_gcs,
      mock_retrieve_retrieve_dicom_from_gcs,
      unused_process,
      request,
      expect_tokenization,
      expect_elixr_c_call,
      expected_output_keys,
      mock_retrieve_instance_from_dicom_store_calls=0,
      mock_retrieve_file_bytes_from_gcs_calls=0,
      mock_retrieve_dicom_from_gcs_calls=0,
  ):

    # Mock model calls - We'll assert that they were called with the correct
    # inputs below.
    mocked_elixr_c_output = np.array([[1]])
    mock_model_runner.run_model.return_value = mocked_elixr_c_output

    mocked_tokenizer_output = {
        "input_word_ids": np.ones([1, 128], dtype=np.int32),
        # Mask of zeros - this will turn into padding of 1s, since padding is
        # (1 - mask).
        "input_mask": np.zeros([1, 128], dtype=np.float32),
    }
    mock_qformer_output = {
        "all_contrastive_img_emb": np.array([[[4]]]),
        "img_emb": np.array([[[5]]]),
        "contrastive_txt_emb": np.array([[[6]]]),
    }
    # Filter outputs for expected keys.
    # We ensure the correct keys are requested in the mock_model_runner call
    # below.
    mock_qformer_output = {
        key: mock_qformer_output[key]
        for key in mock_qformer_output
        if key in expected_output_keys
    }

    if expect_tokenization:
      mock_model_runner.run_model_multiple_output.side_effect = [
          mocked_tokenizer_output,
          mock_qformer_output,
      ]
    else:
      mock_model_runner.run_model_multiple_output.side_effect = [
          mock_qformer_output,
      ]

    response = predictor.Predictor().predict(
        request=request,
        cxr_model_runner=mock_model_runner,
    )

    # Check expected retrieval calls were made
    self.assertEqual(
        mock_retrieve_retrieve_dicom_from_gcs.call_count,
        mock_retrieve_dicom_from_gcs_calls,
    )
    self.assertEqual(
        mock_retrieve_file_bytes_from_gcs.call_count,
        mock_retrieve_file_bytes_from_gcs_calls,
    )
    self.assertEqual(
        mock_retrieve_instance_from_dicom_store.call_count,
        mock_retrieve_instance_from_dicom_store_calls,
    )

    # Check elixr_c model call
    if expect_elixr_c_call:
      mock_model_runner.run_model.assert_called_once_with(
          model_input={
              "input_example": np.array(
                  [_TEST_IMAGE_EXAMPLE.SerializeToString()]
              )
          },
          model_name="elixr_c",
          model_output_key="feature_maps_0",
      )
    else:
      mock_model_runner.run_model.assert_not_called()

    expected_run_model_multiple_output_calls = []
    # Check if tokenizer was called
    if expect_tokenization:
      expected_run_model_multiple_output_calls.append(
          mock.call(
              model_input={
                  "input_1": np.array(["fake_prompt_query"]),
              },
              model_name="tokenizer",
              model_output_keys={"input_word_ids", "input_mask"},
          )
      )

    # Check args for qformer call.
    # These should be trivial for unused types (e.g., elixr_c output in
    # text-only queries), and use the mocked output for used types.
    expected_image_feature = (
        mocked_elixr_c_output
        if expect_elixr_c_call
        else np.zeros([1, 8, 8, 1376], dtype=np.float32).tolist()
    )
    expected_ids = (
        np.ones([1, 1, 128], dtype=np.int32)
        if expect_tokenization
        else np.zeros([1, 1, 128], dtype=np.int32)
    )
    expected_paddings = (
        np.ones([1, 1, 128], dtype=np.float32)
        if expect_tokenization
        else np.zeros([1, 1, 128], dtype=np.float32)
    )

    expected_run_model_multiple_output_calls.append(
        mock.call(
            model_input={
                "image_feature": expected_image_feature,
                "ids": ArrayThatIsEqualTo(expected_ids),
                "paddings": ArrayThatIsEqualTo(expected_paddings),
            },
            model_name="qformer",
            model_output_keys=expected_output_keys,
        )
    )
    mock_model_runner.run_model_multiple_output.assert_has_calls(
        expected_run_model_multiple_output_calls
    )

    # No other calls.
    self.assertEqual(
        mock_model_runner.run_model_multiple_output.call_count,
        2 if expect_tokenization else 1,
    )

    # Check the overall response.
    # Since we're mocking out the qformer call, this really is only testing that
    # the qformer output is packaged into a "predictions" list.
    expected_response = {}
    if "contrastive_txt_emb" in mock_qformer_output:
      expected_response["contrastive_txt_emb"] = mock_qformer_output[
          "contrastive_txt_emb"
      ][0]
    if "all_contrastive_img_emb" in mock_qformer_output:
      expected_response["contrastive_img_emb"] = mock_qformer_output[
          "all_contrastive_img_emb"
      ][0]
    if "img_emb" in mock_qformer_output:
      expected_response["general_img_emb"] = mock_qformer_output["img_emb"][0]

    # Check that the response is as expected.
    # Note that we do a key-by-key comparison with a tolist() comparison,
    # since numpy doesn't consider shape to be part of equality.
    for key, value in response["predictions"][0].items():
      self.assertEqual(value, expected_response[key].tolist())

  # Test for multiple instances.
  # This assumes the processing of individual instances is sufficiently tested
  # in the above suite, and tests that the multiple instances are processed
  # in order and packaged into the output.
  @mock.patch.object(model_runner, "ModelRunner", autospec=True)
  def test_predict_with_multiple_instances_success(self, mock_model_runner):
    request = {
        "instances": [
            {"prompt_query": "fake_prompt_query_1"},
            {"prompt_query": "fake_prompt_query_2"},
        ]
    }
    mocked_tokenizer_output_1 = {
        "input_word_ids": np.ones([1, 128], dtype=np.int32),
        # Mask of zeros - this will turn into padding of 1s, since padding is
        # (1 - mask).
        "input_mask": np.zeros([1, 128], dtype=np.float32),
    }

    mock_qformer_output_1 = {
        "contrastive_txt_emb": np.array([[6]]),
    }

    mocked_tokenizer_output_2 = {
        "input_word_ids": np.full([1, 128], 2, dtype=np.int32),
        # Mask of zeros - this will turn into padding of 1s, since padding is
        # (1 - mask).
        "input_mask": np.zeros([1, 128], dtype=np.float32),
    }
    mock_qformer_output_2 = {
        "contrastive_txt_emb": np.array([[9]]),
    }

    mock_model_runner.run_model_multiple_output.side_effect = [
        mocked_tokenizer_output_1,
        mock_qformer_output_1,
        mocked_tokenizer_output_2,
        mock_qformer_output_2,
    ]

    response = predictor.Predictor().predict(
        request=request,
        cxr_model_runner=mock_model_runner,
    )

    mock_model_runner.run_model_multiple_output.assert_has_calls([
        mock.call(
            model_input={
                "input_1": np.array(["fake_prompt_query_1"]),
            },
            model_name="tokenizer",
            model_output_keys={"input_word_ids", "input_mask"},
        ),
        mock.call(
            model_input={
                "image_feature": mock.ANY,
                "ids": ArrayThatIsEqualTo(
                    [mocked_tokenizer_output_1["input_word_ids"]]
                ),
                "paddings": ArrayThatIsEqualTo(
                    np.ones([1, 1, 128], dtype=np.float32)
                ),
            },
            model_name="qformer",
            model_output_keys={
                "contrastive_txt_emb",
            },
        ),
        mock.call(
            model_input={
                "input_1": np.array(["fake_prompt_query_2"]),
            },
            model_name="tokenizer",
            model_output_keys={"input_word_ids", "input_mask"},
        ),
        mock.call(
            model_input={
                "image_feature": mock.ANY,  # expected_image_feature,
                "ids": ArrayThatIsEqualTo(
                    [mocked_tokenizer_output_2["input_word_ids"]]
                ),
                "paddings": ArrayThatIsEqualTo(
                    np.ones([1, 1, 128], dtype=np.float32)
                ),
            },
            model_name="qformer",
            model_output_keys={
                "contrastive_txt_emb",
            },
        ),
    ])

    # No other calls.
    self.assertEqual(mock_model_runner.run_model_multiple_output.call_count, 4)

    # Check the overall response.
    # Since we're mocking out the qformer call, this really is only testing that
    # the qformer outputs are packaged into a "predictions" list and then
    # returned.
    self.assertEqual(
        response,
        {"predictions": [mock_qformer_output_1, mock_qformer_output_2]},
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="gcs_and_bytes",
          request={
              "instances": [{
                  "gcs_uri": "fake_dicom.dcm",
                  "input_bytes": b"some_bytes",
              }]
          },
      ),
      dict(
          testcase_name="bytes_and_dicomweb",
          request={
              "instances": [{
                  "input_bytes": b"some_bytes",
                  "dicomweb_uri": "fake_dicom_web_uri",
              }]
          },
      ),
  )
  @mock.patch.object(model_runner, "ModelRunner", autospec=True)
  def test_predict_with_multiple_image_data_types_returns_error(
      self, mock_model_runner, request
  ):
    response = predictor.Predictor().predict(
        request=request,
        cxr_model_runner=mock_model_runner,
    )

    self.assertLen(response["predictions"], 1)
    self.assertLen(response["predictions"][0], 1)
    self.assertIn(
        "Multiple image keys found",
        response["predictions"][0]["error"]["description"],
    )

  @mock.patch.object(model_runner, "ModelRunner", autospec=True)
  def test_predict_with_some_valid_and_some_invalid(self, mock_model_runner):
    request = {
        "instances": [
            {"gcs_uri": "fake_dicom.dcm", "input_bytes": b"some_bytes"},
            {"prompt_query": "fake_prompt_query_2"},
            {"gcs_uri": "fake_dicom.dcm", "input_bytes": b"some_bytes"},
            {"prompt_query": "fake_prompt_query_2"},
        ]
    }

    mocked_tokenizer_output = {
        "input_word_ids": np.ones([1, 128], dtype=np.int32),
        # Mask of zeros - this will turn into padding of 1s, since padding is
        # (1 - mask).
        "input_mask": np.zeros([1, 128], dtype=np.float32),
    }

    mock_qformer_output_1 = {
        "all_contrastive_img_emb": np.array([[[4]]]),
        "img_emb": np.array([[5]]),
        "contrastive_txt_emb": np.array([[6]]),
    }
    mock_qformer_output_2 = {
        "all_contrastive_img_emb": np.array([[[7]]]),
        "img_emb": np.array([[8]]),
        "contrastive_txt_emb": np.array([[9]]),
    }

    mock_model_runner.run_model_multiple_output.side_effect = [
        mocked_tokenizer_output,
        mock_qformer_output_1,
        mocked_tokenizer_output,
        mock_qformer_output_2,
    ]

    response = predictor.Predictor().predict(
        request=request,
        cxr_model_runner=mock_model_runner,
    )

    err_msg_substr = "Multiple image keys found in request instance"
    self.assertLen(response["predictions"], 4)
    self.assertIn(
        err_msg_substr, response["predictions"][0]["error"]["description"]
    )
    self.assertEqual(
        {
            "contrastive_img_emb": np.array([[4]]),
            "general_img_emb": np.array([[5]]),
            "contrastive_txt_emb": np.array([[6]]),
        },
        response["predictions"][1],
    )
    self.assertIn(
        err_msg_substr, response["predictions"][2]["error"]["description"]
    )
    self.assertEqual(
        {
            "contrastive_img_emb": np.array([[7]]),
            "general_img_emb": np.array([[8]]),
            "contrastive_txt_emb": np.array([[9]]),
        },
        response["predictions"][3],
    )


if __name__ == "__main__":
  absltest.main()
