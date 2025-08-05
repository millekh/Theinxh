# Theinxh

This repository includes an example Streamlit dashboard for exploring curvature
simulations and simple multi‑agent interactions powered by LangChain and
Hugging Face models. Temporal curvature ``Λ`` is computed using the TICE
equation and a lightweight multi‑agent extension. The dashboard can simulate
randomised agents or derive curvature directly from MNIST digit samples to
illustrate behaviour on real data.

Run the demo with:

```bash
pip install streamlit langchain langchain-experimental huggingface_hub
streamlit run qbond_curvature_dashboard.py
```

Curvature utilities are available in ``tice.py`` and ``mnist_curvature.py`` for
reuse in other projects.

## FastAPI Microservice

Metrics can also be accessed programmatically via a FastAPI service exposing
three endpoints:

- ``POST /simulate/curvature`` – generate random multi‑agent simulations and
  return ``Λ`` and curve index ``C`` values.
- ``POST /compute/xi_chi`` – compute the ``Ξχ`` negentropy metric for a
  probability distribution.
- ``POST /forecast/scg`` – forecast Symbolic Curvature Gain ``SCG`` from a
  series of curvature scores.

Start the service with:

```bash
pip install fastapi uvicorn
uvicorn fastapi_service:app --reload
```

### Plugin Connectors

``connectors.py`` contains lightweight helper functions for integrating the
service with external frameworks:

- LangChain tools can invoke curvature simulations directly.
- Groq API style callables for accelerator driven workloads.
- Hugging Face Spaces demo hooks for community apps.
- AutoGPT plugin helpers to monitor ``SCG`` during swarms.

These connectors demonstrate how the microservice can plug into different AI
stacks.
