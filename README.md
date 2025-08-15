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

### MNIST curvature metrics

For experiments on real MNIST data with an Ollivier–Ricci cross‑check, install
the optional dependency and run the metrics script:

```bash
pip install GraphRicciCurvature networkx
python mnist_metrics.py
```

The script samples MNIST digits, computes multi‑agent ``Λ`` values with a
``reg_beta`` of ``0.02``, checks proof coherence against the mean Ricci
curvature, and reports the average memory compression ``Ω``.

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

## Quantum Sentinel Simulation

For a larger multi-agent playground complete with optional FastAPI and dashboard
generation, run the quantum sentinel script:

```bash
pip install torch
python tice_multi_agent_sim_quantum_sentinel_plus.py
```

The script trains small convolutional agents on synthetic images, computes
advanced curvature metrics (Forman and Ollivier–Ricci), forecasts future
``Λ`` values, and can export a dashboard image summarising the run.

## Final Presentation Simulation

For compliance-focused demos with adaptive difficulty and secure logging run:

```bash
python tice_multi_agent_sim_final_presentation.py
```

This variant encrypts trust matrices when ``cryptography`` is installed and adapts agent counts based on curvature and adversarial signals.
