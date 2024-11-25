# Chest X-Ray Foundation

CXR Foundation is a machine learning (ML) model that produces embeddings based
on images of chest X-rays. The embeddings can be used to
[efficiently build AI models](https://en.wikipedia.org/wiki/Transfer_learning)
for chest X-ray related tasks, requiring less data and less compute than having
to fully train a model without the embeddings.

The model has been optimized for chest X-rays, but researchers have reported
success using it for other types of X-rays, including X-rays of other body parts
and even veterinary X-rays.

As a Health AI Developer Foundations (HAI-DEF) model trained on large scale
datasets, CXR Foundation helps businesses and institutions in healthcare and
life sciences do more with their chest X-ray data with less data, accelerating
their ability to build AI models for chest X-ray image analysis.

## Get started

*   Read our
    [developer documentation](https://developers.google.com/health-ai-developer-foundations/cxr-foundation/get-started)
    to see the full range of next steps available, including learning more about
    the model through its
    [model card](https://developers.google.com/health-ai-developer-foundations/cxr-foundation/model-card)
    or
    [serving API](https://developers.google.com/health-ai-developer-foundations/cxr-foundation/serving-api).

*   Explore this repository, which contains [notebooks](./notebooks) for using
    the model from Hugging Face and Vertex AI as well as the
    [implementation](./serving) of the container that you can deploy to Vertex
    AI.

*   Visit the model on
    [Hugging Face](https://huggingface.co/google/cxr-foundation) or
    [Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/cxr-foundation).

## Contributing

We are open to bug reports, pull requests (PR), and other contributions. See
[CONTRIBUTING](CONTRIBUTING.md) and
[community guidelines](https://developers.google.com/health-ai-developer-foundations/community-guidelines)
for details.

see [CONTRIBUTING](CONTRIBUTING.md) for details.

## License

While the model is licensed under the
[Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms),
everything in this repository is licensed under the Apache 2.0 license, see
[LICENSE](LICENSE).
