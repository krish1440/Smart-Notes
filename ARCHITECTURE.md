# 🏗️ Smart Academic Notes - System Architecture

## 1. Executive Overview
Smart Academic Notes is a high-performance, AI-driven synthesis engine designed to transform unstructured academic content (PDFs, Audio) into structured, searchable knowledge. The system leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide context-aware insights while maintaining strict data isolation.

---

## 2. Technical Stack
The platform is built on a modern, distributed architecture:

- **Logic Layer**: Flask (Python 3.12) utilizing deferred initialization for optimized startup.
- **Identity Provider**: Supabase Auth (JWT-based session management).
- **Intelligence Engine**: 
    - **LLM**: Google Gemini 2.5 Flash for high-speed summarization.
    - **Embeddings**: Hugging Face `all-MiniLM-L6-v2` for semantic representation.
- **Storage Layer**:
    - **Relational**: Supabase (PostgreSQL).
    - **Vector**: Pinecone (Serverless) for high-dimensional similarity search.
    - **Blob**: Cloudinary for raw media assets.

---

## 3. Data Ingestion Pipeline
The core value proposition lies in our multi-stage ingestion pipeline:

### Phase A: Extraction
1. **Media Ingestion**: PDFs and audio files are validated for size-quota limits.
2. **Raw Processing**:
    - **PDF**: PyMuPDF (`fitz`) extracts raw text while preserving layout hints.
    - **Audio**: (Planned/Future) Whisper-based transcription of lecture notes.

### Phase B: Transformation
1. **Semantic Chunking**: Text is split using `RecursiveCharacterTextSplitter` with balanced overlap to ensure context preservation across boundaries.
2. **Vectorization**: Chunks are transformed into 384-dimensional dense vectors via Hugging Face.

### Phase C: Indexing
1. **Namespace isolation**: Every document is indexed in a unique Pinecone namespace (`user_{id}_note_{id}`) to prevent cross-pollination of data and ensure privacy.

---

## 4. Query & RAG Workflow
When a user interacts with a document:
1. **Vector Search**: The user's query is vectorized and compared against the document's namespace in Pinecone.
2. **Context Retrieval**: Top-K matches are retrieved and injected into the LLM prompt.
3. **Synthesis**: Google Gemini generates a response grounded strictly in the retrieved context.

---

## 5. Security & Rate Limiting
- **Authentication**: All API endpoints (except Auth) require a Bearer JWT.
- **Quota Management**: A dedicated middleware monitors Supabase to enforce a strict limit of 5 PDF uploads per 24 hours per user.
- **API Resilience**: The system implements an **Exponential Backoff** strategy (`rate_limit_llm`) to handle Google AI platform rate limits gracefully.

---

## 6. Infrastructure Setup
Refer to the [Developer Setup Guide](./SETUP.md) for environment initialization and service configuration.
