# Nova AI Customer Support RAG

A semantic Retrieval-Augmented Generation (RAG) customer support assistant built with Python, OpenAI embeddings, and an LLM.

## Overview

Nova AI Customer Support uses semantic search to retrieve relevant information from a small customer-support knowledge base before generating an answer.

Instead of relying entirely on the language model's internal knowledge, Nova retrieves relevant information from the knowledge base and uses that information as context when answering customer questions.

## How It Works

The system follows this pipeline:

Customer Question
        ↓
Create Question Embedding
        ↓
Semantic Similarity Search
        ↓
Retrieve Top Relevant Documents
        ↓
Apply Similarity Threshold
        ↓
Build RAG Prompt
        ↓
Generate Answer
        ↓
Return Response

## Features

- Semantic document retrieval
- OpenAI text embeddings
- Cosine similarity search
- Top-K document retrieval
- Similarity threshold to reduce irrelevant answers
- Saved document embeddings for efficiency
- Interactive customer-support conversation
- Knowledge-grounded responses
- Fallback response when information is not available

## Knowledge Base

The current knowledge base contains information about:

- Return policy
- Shipping and delivery
- Product warranty
- Payment methods

Example:

**Customer:** How long does shipping take?

**Nova:** Standard shipping takes 3-5 business days.

## Example

```text
Customer: Does the camera have a warranty?

Retrieved document:
warranty_policy

Similarity score:
0.6739

Nova's answer:
Yes, the Nova Camera comes with a 2-year warranty.