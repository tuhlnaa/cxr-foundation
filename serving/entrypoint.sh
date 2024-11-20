#!/bin/bash
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

# This script launches the CXR foundation prediction container.
# It copies the model from Google Cloud Storage (GCS) to a local directory
# and launches the TensorFlow Model Server and the CXR foundation server.
# The script waits for any process to exit and then exits with the status of
# the process that exited first.

set -e

# Port for the TensorFlow Model Server REST API.
export MODEL_REST_PORT=8600

echo "Obtaining models"

# Copy model from Google Cloud Storage (GCS) to local directory
# Doesn't work parallelized - use improved downloader script?
mkdir -p "/model/elixr_c/1"
mkdir -p "/model/qformer/1"
mkdir -p "/model/tokenizer/1"

# TODO(b/379159076): Remove gcloud
gcloud storage cp "${AIP_STORAGE_URI}/elixr-c-v2-pooled/*" "/model/elixr_c/1" --recursive
gcloud storage cp "${AIP_STORAGE_URI}/pax-elixr-b-text/*" "/model/qformer/1" --recursive
gcloud storage cp "${AIP_STORAGE_URI}/tokenizer/*" "/model/tokenizer/1" --recursive

echo "Prediction container start, launching model server"

/usr/bin/tensorflow_model_server \
    --port=8500 \
    --rest_api_port="$MODEL_REST_PORT" \
    --model_config_file=model_config.txtpb \
    --xla_cpu_compilation_enabled=true &
export MODEL_PID=$!

echo "Launching front end"

/server-env/bin/python3.12 server_gunicorn.py --alsologtostderr \
    --verbosity=1 &
export SERVER_PID=$!

ps

# Wait for any process to exit
echo "Container persisting until tfserving $MODEL_PID or server $SERVER_PID ends."
wait -n $MODEL_PID $SERVER_PID
export EXIT_CODE=$?

echo "Exiting due to process termination"

ps

# Exit with status of process that exited first
exit $EXIT_CODE
