# Distributed GPT-2 LLM Inference Service

A scalable **Large Language Model (LLM) inference system** built using **FastAPI, Redis, PyTorch, and Docker**.

This project implements the **GPT-2 architecture from scratch in PyTorch**, then loads pretrained GPT-2 weights to perform inference through an asynchronous microservice system.

The system accepts text prompts via an API, queues them in Redis, and processes them using a worker service running the GPT-2 model.

---

# Key Highlights

• Implemented **GPT-2 transformer architecture from scratch** using PyTorch
• Loaded **pretrained GPT-2 weights** for inference
• Built **asynchronous LLM inference system** using Redis queues
• Designed **worker-based model execution pipeline**
• Containerized full system using Docker Compose
• Implemented task-based API endpoints for scalable inference

---

# GPT-2 Architecture Implementation

The GPT-2 model was implemented manually in PyTorch, including:

• Token + positional embeddings
• Multi-Head Self Attention
• Causal masking for autoregressive generation
• Transformer blocks with residual connections
• Layer normalization
• Feed-forward MLP layers
• Output projection to vocabulary logits

After constructing the architecture, **pretrained GPT-2 weights were loaded into the custom implementation** to perform inference.

This approach demonstrates a deeper understanding of **transformer internals and large language model architecture**.

---

# Architecture

Client
↓
FastAPI API
↓
Redis Queue
↓
Worker Service
↓
Custom GPT-2 Model (PyTorch)
↓
Generated Text Stored in Redis

---

# Tech Stack

**Backend**

* FastAPI
* Python

**Machine Learning**

* PyTorch
* Custom GPT-2 architecture implementation

**Queue System**

* Redis

**Infrastructure**

* Docker
* Docker Compose

---

# API Endpoints

### Generate Text

POST `/generate`

Request

```json
{
  "prompt": "Large language models are",
  "max_tokens": 100
}
```

Response

```json
{
  "task_id": "uuid",
  "status": "queued"
}
```

---

### Retrieve Result

GET `/result/{task_id}`

Response

```json
{
  "task_id": "uuid",
  "status": "completed",
  "result": "Generated text..."
}
```

---

# Running the Project

Clone the repository

```
git clone https://github.com/yourusername/gpt-llm-service.git
cd gpt-llm-service
```

Start services

```
docker compose up --build
```

API Documentation

```
http://localhost:8000/docs
```

---

# Learning Outcomes

This project demonstrates practical understanding of:

• Transformer architecture implementation
• GPT-2 weight loading and inference
• Asynchronous ML inference systems
• Queue-based distributed architecture
• Containerized ML deployment with Docker

---

# Future Improvements

• Streaming text generation
• GPU optimized workers
• Batch inference support
• Model scaling and load balancing

---
