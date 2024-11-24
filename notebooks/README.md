# CXR Foundation Notebooks

*   [Quick start with Hugging Face](quick_start_with_hugging_face.ipynb) -
    Example of encoding a chest x-ray (CXR) image and/or text prompt into an
    embedding vector by running the model locally from Hugging Face.

*   [Quick start with Vertex Model Garden](quick_start_with_model_garden.ipynb) -
    Example of serving the model on
    [Vertex AI](https://cloud.google.com/vertex-ai/docs/predictions/overview)
    and using Vertex AI APIs to encode CXR images and/or text prompts to
    embeddings in online or batch workflows.

*   [Train a data efficient classifier](train_data_efficient_classifier.ipynb) -
    Example of using the generated embeddings to train a custom classifier with
    less data and compute.

*   [Retrieve images by text queries](retrieve_images_by_text.ipynb) - Example
    of using the generated embeddings for retrieving images using text queries,
    leveraging text-image similarity.

*   [Classify images with natural language](classify_images_with_natural_language.ipynb) -
    Example of using the natural language text embeddings to classify CXR images.