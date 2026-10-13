# Source-Grounded RAG Chatbot for an Academic Learner Guide

**A retrieval-augmented AI assistant designed to answer questions from one approved document, show supporting sources and abstain when evidence is insufficient.**

## Overview

This project develops a **Retrieval-Augmented Generation (RAG)** chatbot for the EDAB6808 Learner Guide. Unlike an unrestricted chatbot, it is designed to use **only retrieved passages from the guide** to answer questions and show their page and section references.

The prototype is built in a Google Colab notebook and presented through a **Gradio** interface.

## Architecture

1. **Document ingestion:** PyMuPDF extracts and cleans learner-guide text.
2. **Chunking:** section-aware chunks of approximately 200 tokens preserve page and section metadata.
3. **Embeddings:** `BAAI/bge-small-en-v1.5` produces 384-dimensional vectors.
4. **Vector database:** **Qdrant Local** stores embeddings and text metadata, with rerun-safe indexing and optional Google Drive persistence.
5. **Retrieval:** dense retrieval is combined with **BM25**, **Reciprocal Rank Fusion** and cross-encoder reranking.
6. **Answer generation:** **Qwen2.5-3B-Instruct** generates an answer from retrieved evidence only.
7. **Trust controls:** citation verification, an evidence-support proxy and explicit abstention for unsupported questions.
8. **Interface:** a Gradio chat experience displays both answers and retrieved evidence.

## Evaluation design

The notebook tests **20 questions**: 14 direct factual, three multi-part and three unsupported queries. It calculates retrieval measures, answer and citation accuracy, abstention performance, latency and a **proxy** for faithfulness. Results should be read with the limitations in the notebook; no automated score guarantees that every statement is true.

**Model note:** Qwen is used as a released **instruction-tuned** model. This project does **not** claim to fine-tune Qwen on the learner guide.

## Repository contents

- `EDAB6808_Ass5_RAG_Gradio_final.ipynb` — ingestion, chunking, retrieval, generation, evaluation and Gradio interface.

The assignment brief and the original learner guide were included in the submitted ZIP but should only be published if redistribution is permitted by the university. Otherwise, keep the PDF private and let authorised users supply their own copy.

## Running the prototype

1. Open the notebook in **Google Colab** and select a **T4 GPU** runtime.
2. Provide an authorised copy of the learner-guide PDF in the expected Google Drive project folder or upload it when prompted.
3. Run the setup cell. The notebook pins packages including `pymupdf`, `qdrant-client`, `sentence-transformers`, `transformers`, `rank-bm25`, `gradio` and `accelerate`.
4. Run the remaining cells in order. The first run downloads open-weight models and builds the Qdrant index; later runs can reuse the saved collection.
5. Launch the Gradio app in the final section, then test factual and unsupported queries.

## Limitations

A small evaluation set cannot establish production-level reliability. The citation and faithfulness checks are heuristic, retrieved text can be incomplete and the system should not be treated as a general-purpose source of university policy beyond the supplied guide.

## Contributors

**Lwazi Nhlapo**, **Boipelo Letwaba** and **Buhle Mbokodo** — collaborative academic project.
