# 📡 API Reference Manual

This manual provides detailed specifications for the Smart Academic Notes REST API. All requests, except for the authentication branch, require a valid `Authorization: Bearer <JWT>` header.

---

## 1. Authentication (`/api/auth`)

### POST `/signup`
Initiates a new user registration.
- **Request Body**: `{ "email": "str", "password": "str" }`
- **Response (200)**: `{ "message": "OTP sent to your email", "email": "str" }`
- **Notes**: Triggers a Supabase verification email.

### POST `/verify-otp`
Validates the email OTP and finalizes registration.
- **Request Body**: `{ "otp": "str" }`
- **Response (200)**: `{ "token": "JWT", "user": { ... } }`

### POST `/login`
Authenticates a user and issues a session token.
- **Request Body**: `{ "email": "str", "password": "str" }`
- **Response (200)**: `{ "token": "JWT", "user": { ... } }`

---

## 2. Document Management (`/api/upload`)

### POST `/pdf`
Processes a new research PDF and triggers the AI pipeline.
- **Content-Type**: `multipart/form-data`
- **Fields**: `file` (Binary)
- **Response (200)**: `{ "message": "PDF uploaded and processed", "note_id": "UUID" }`
- **Constraints**: 5MB limit; 5 uploads per 24h.

---

## 3. Intelligence & Interaction (`/api/chat`)

### POST `/`
Performs a semantic query against a specific document.
- **Request Body**: `{ "note_id": "UUID", "message": "str" }`
- **Response (200)**: `{ "answer": "str", "context": [...] }`
- **Engine**: RAG (Pinecone + Gemini)

### GET `/history/<note_id>`
Retrieves the contextual memory of a specific conversation.
- **Response (200)**: `[ { "role": "user", "msg": "..." }, ... ]`

---

## 4. Notes & Export

### GET `/notes`
Returns a collection of all notes owned by the authenticated user.

### GET `/export/pdf/<note_id>`
Generates a downloadable high-end PDF summary and chat history.
- **Response (200)**: Binary PDF Stream.
