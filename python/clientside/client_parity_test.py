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
"""Internal test file for testing clients.

This ensures that the client APIs work as expected, and that the two different
clients (Hugging Face and Vertex) return the same embeddings.
TODO(nickgeorge): Turn this into a proper test that can be open sourced.
"""

from collections.abc import Sequence
import io
from absl import app
import numpy as np
from PIL import Image
import requests
from clientside import clients

_DICOMWEB_URL = "https://proxy.imaging.datacommons.cancer.gov/current/viewer-only-no-downloads-see-tinyurl-dot-com-slash-3j3d9jyp/dicomWeb/studies/1.3.6.1.4.1.14519.5.2.1.6279.6001.570861972342030356759331781072/series/1.3.6.1.4.1.14519.5.2.1.6279.6001.778504284378789172971189645726/instances/1.3.6.1.4.1.14519.5.2.1.6279.6001.740389520167988096048969742405"
_GCS_DICOM_URL = "gs://cxr-foundation-demo/cxr14/inputs/00026260_000.dcm"
_GCS_PNG_URL = "gs://hai-cd3-use-case-dev-hai-def-test-data-vault-entry/cxr-foundation/png-files/newcxr.png"
_REMOTE_PNG_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Chest_Xray_PA_3-8-2010.png/1024px-Chest_Xray_PA_3-8-2010.png"


def _text_embeddings_equal(
    a: clients.TextEmbedding,
    b: clients.TextEmbedding,
) -> bool:
  return np.allclose(
      np.array(a.contrastive_txt_emb),
      np.array(b.contrastive_txt_emb),
      atol=1e-6,
  )


def _image_embeddings_equal(
    a: clients.ImageEmbedding,
    b: clients.ImageEmbedding,
) -> bool:
  return np.allclose(
      np.array(a.contrastive_img_emb),
      np.array(b.contrastive_img_emb),
      atol=1e-4,
  ) and np.allclose(
      np.array(a.general_img_emb),
      np.array(b.general_img_emb),
      atol=1e-4,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  vertex_client = clients.VertexCxrClient(
      endpoint_name="130", project="4016704501", location="us-west1"
  )
  hugging_face_client = clients.make_hugging_face_client("/tmp/huggingface")

  # Text Embeddings
  hf_text_embs = hugging_face_client.get_text_embeddings(
      ["one", "two", "three"]
  )
  vertex_text_embs = vertex_client.get_text_embeddings(["one", "two", "three"])
  assert _text_embeddings_equal(hf_text_embs[0], vertex_text_embs[0])
  assert _text_embeddings_equal(hf_text_embs[1], vertex_text_embs[1])
  assert _text_embeddings_equal(hf_text_embs[2], vertex_text_embs[2])

  # Image Embeddings from DICOMweb
  hf_dicomweb_emb = hugging_face_client.get_image_embeddings_from_dicomweb(
      [_DICOMWEB_URL]
  )
  vertex_dicomweb_emb = vertex_client.get_image_embeddings_from_dicomweb(
      [_DICOMWEB_URL]
  )
  assert _image_embeddings_equal(hf_dicomweb_emb[0], vertex_dicomweb_emb[0])

  hf_dicom_from_gcs_emb = hugging_face_client.get_image_embeddings_from_gcs(
      [_GCS_DICOM_URL]
  )
  vertex_dicom_from_gcs_emb = vertex_client.get_image_embeddings_from_gcs(
      [_GCS_DICOM_URL]
  )
  assert _image_embeddings_equal(
      hf_dicom_from_gcs_emb[0], vertex_dicom_from_gcs_emb[0]
  )
  # Image Embeddings from PNG in GCS
  hf_png_from_gcs_emb = hugging_face_client.get_image_embeddings_from_gcs(
      [_GCS_PNG_URL]
  )
  vertex_png_from_gcs_emb = vertex_client.get_image_embeddings_from_gcs(
      [_GCS_PNG_URL]
  )
  assert _image_embeddings_equal(
      hf_png_from_gcs_emb[0], vertex_png_from_gcs_emb[0]
  )

  # Local Image Embeddings
  headers = {"User-Agent": "clients-parity-test"}
  response = requests.get(
      _REMOTE_PNG_URL,
      headers=headers,
  )
  img = Image.open(io.BytesIO(response.content))

  hf_png_from_local_image = (
      hugging_face_client.get_image_embeddings_from_images([img])
  )
  vertex_png_from_local_image = vertex_client.get_image_embeddings_from_images(
      [img]
  )
  assert _image_embeddings_equal(
      hf_png_from_local_image[0], vertex_png_from_local_image[0]
  )

  print("All tests passed!")


if __name__ == "__main__":
  app.run(main)
