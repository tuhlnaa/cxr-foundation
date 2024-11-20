# CXR Foundation Serving

This is a container implementation that can serve the model and is used for
Vertex AI serving.

## Description for select files and folders

*   [`Dockerfile`](./Dockerfile): This file defines the Docker image that will
    be used to serve the model. It includes the necessary dependencies, such as
    TensorFlow and the model server.
*   [`requirements.txt`](./requirements.txt): This file lists the Python
    packages that are required to run the model server.
*   [`server.py`](./server.py): This file contains the code that runs the model
    server. It loads the model from a specified location and then starts the
    server.
*   [`start_server.sh`](./start_server.sh): This file is a shell script that
    builds the Docker image and then runs the model server.
*   [`test_client.py`](./test_client.py): This file contains code that can be
    used to test the model server. It sends a request to the server and prints
    the response.
*   [`Data Processing`](./data_processing): This folder contains code for
    processing the CXR foundation dataset.
*   [`prediction_container`](./prediction_container): This folder contains code
    for a container that can be used to serve predictions from any model.
