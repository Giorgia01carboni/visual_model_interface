# SpatialLM Industrial Assistant

A specialized GUI designed to ground natural language queries into 3D industrial point clouds. This tool wraps **SpatialLM** to allow users to upload `.ply` scenes, ask natural language questions (e.g., "Where are the valves?"), and visualize 3D detection results interactively.

> **Note**: This instance is configured to run with a custom model fine-tuned on industrial facility datasets.

## Features

* **3D Visualization**: High-performance interactive viewer built with **Plotly**. Features smart subsampling and client-side toggles for bounding box visibility.
* **LLM Orchestration**: Integrated **GPT-4o** (via `LLMService`) to parse user intent and control the 3D scanning tool.
* **Dynamic Inference**: Adjust model parameters (`top_k`, `temperature`, `beams`) on the fly without restarting.
* **Reactive UI**: Built with **Gradio** for seamless state management between chat and 3D visuals.

## About the Model

This project leverages **SpatialLM**, a spatial reasoning model capable of understanding and grounding language in 3D environments.

While the core architecture is based on SpatialLM, this assistant is designed to load custom checkpoints trained for specific detection tasks (e.g., pipes, boilers, electrical equipment) in complex industrial layouts.

## Installation

1. **Install Core Model**:
   Ensure you have the local `spatiallm` package installed (refer to the original SpatialLM repository for environment setup).

2. **Install Assistant Dependencies**:
   ```bash
   pip install -r requirements.txt

## Usage

1. **Set API Keys**:
   ```bash
   export OPENAI_API_KEY="your-key..."
## Credits & Acknowledgements

This interface is built upon the **SpatialLM** research project. If you use the core model architecture, please consider citing the original authors:

> **SpatialLM: Training Large Language Models for Structured Indoor Modeling**
> *Mao, Yongsen et al. (NeurIPS 2025)*
> [Original Repository](https://github.com/manycore-research/SpatialLM) | [ArXiv Paper](https://arxiv.org/abs/2506.07491)

```bibtex
@inproceedings{SpatialLM,
  title     = {SpatialLM: Training Large Language Models for Structured Indoor Modeling},
  author    = {Mao, Yongsen and Zhong, Junhao and Fang, Chuan and Zheng, Jia and Tang, Rui and Zhu, Hao and Tan, Ping and Zhou, Zihan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
