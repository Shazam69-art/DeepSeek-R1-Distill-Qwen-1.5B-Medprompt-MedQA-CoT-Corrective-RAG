# MedRAG-CRAG: Corrective Retrieval-Augmented Generation for Medical Question Answering

## Overview

MedRAG-CRAG is an efficient, open-source Retrieval-Augmented Generation (RAG) system tailored for medical question answering. It provides evidence-based, step-by-step responses by retrieving relevant Chain-of-Thought (CoT) reasoning from a specialized knowledge base.

The system is lightweight and designed to run on standard consumer hardware (CPU or GPU), making it accessible for research, education, and prototyping in medical AI applications.

**Key Features**:
- Corrective RAG (CRAG) for robust relevance filtering
- Granular knowledge strips (question, options, reasoning, combined) for precise retrieval
- FAISS-based dense vector search with cosine similarity
- Streaming responses with source citations
- Low memory footprint (~3GB on GPU)

**Disclaimer**: This system is for **educational and research purposes only**. It is not intended for clinical decision-making. Always consult qualified healthcare professionals for medical advice.

## Dataset

The knowledge base is derived from the [Medprompt-MedQA-CoT](https://huggingface.co/datasets/HPAI-BSC/Medprompt-MedQA-CoT) dataset (HPAI-BSC, Apache 2.0 license).

- **Source**: Enhances the USMLE-style MedQA dataset with high-quality CoT reasoning.
- **Generation**: Created using Llama-3.1-70B-Instruct, prompted to analyze options independently and provide detailed justifications.
- **Size**: Approximately 10,000+ Q&A pairs, expanded into ~40,000 knowledge strips.
- **Purpose**: Optimized for RAG to improve contextual reasoning in medical multiple-choice QA.

Related paper: [Pareto-Optimized Open-Source LLMs for Healthcare via Context Retrieval (arXiv:2409.15127)](https://arxiv.org/abs/2409.15127)

## Architecture

### Embedding Generation
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Process**: 
  - Break each Q&A into 4 knowledge strips.
  - Generate normalized embeddings in batches.
  - Build FAISS IndexFlatIP for fast cosine similarity search.
- **Output**: `embeddings.npy`, `knowledge_strips.pkl`, `metadata.pkl`, `config.json`

### Inference (RAG Pipeline)
- **Retriever**:
  - Embed user query using the same MiniLM model.
  - Search FAISS index (top-2k candidates).
  - Apply CRAG: Filter by relevance threshold (≥0.6); fallback to top-k if insufficient.
- **Prompt Construction**: Prioritize "combined" and "reasoning" strips; inject top 3 into prompt.
- **Generator**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (1.5B parameters, FP16 on GPU)
- **Output**: Streaming response with retrieved strip count and source relevance scores.

## Requirements
- Python 3.10+
- Packages: `gradio`, `transformers`, `torch`, `faiss-cpu`, `sentence-transformers`, `numpy`, `tqdm`

## Usage

1. **Generate Embeddings** (Run once):
   - Upload `medprompt_medqa_cot_llama31.json` to Colab `/content/`.
   - Execute the embedding generator script.

2. **Run Inference**:
   - Ensure embedding files are in `/content/`.
   - Execute the Gradio inference script.
   - Access the web interface via the shared link.

## Limitations & Future Work
- Generator model (1.5B) may occasionally hallucinate; larger models can improve accuracy.
- Knowledge base limited to MedQA domain (USMLE-style questions).
- Planned: Integration with larger medical LLMs (e.g., Meditron-7B) and expanded datasets.

## License
Apache 2.0 (aligned with the source dataset).

## Acknowledgments
- HPAI-BSC for the Medprompt-MedQA-CoT dataset.
- Hugging Face for hosting models and datasets.
- DeepSeek AI for the distilled Qwen model.

For questions or contributions, feel free to open an issue!
