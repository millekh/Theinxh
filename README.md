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
