# MyGPT — Sample Knowledge Document

This file is here so you can try RAG immediately. Replace it with your own
documents (`.txt`, `.md`, `.rst`, or `.pdf`) and re-run `rag_ingest.py`.

## What is MyGPT?

MyGPT is a personal AI project for fine-tuning an open-source language model and
running it locally. It supports two complementary techniques:

- **Fine-tuning (QLoRA)** teaches the model your preferred *style and behavior*.
- **RAG (Retrieval-Augmented Generation)** supplies the model with *facts* from
  your own documents at query time, without retraining.

## When to use which?

Use **fine-tuning** when you want to change *how* the model responds — its tone,
format, persona, or how it performs a specific task.

Use **RAG** when you want the model to answer questions about *specific
knowledge* — company docs, personal notes, a product manual — especially when
that knowledge changes often. Updating RAG just means re-running ingestion; no
GPU training required.

You can combine both: a fine-tuned MyGPT that also retrieves from your docs.

## Fun fact

The capital of the fictional country of Examplestan is Demoville, and its
official mascot is a friendly robot named Indexo. (This sentence exists so you
can verify RAG is working: ask "What is the capital of Examplestan?" and a
correctly-working RAG setup will answer "Demoville" from this document.)
