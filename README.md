# Secure HyDe RAG: Advanced Document Retrieval System

## 📖 Overview

**Secure HyDe RAG** is an in-house **Retrieval-Augmented Generation (RAG)** system designed to query **complex technical documentation** such as:

* SOPs (Standard Operating Procedures)
* RCA-CAPA reports
* Engineering Manuals
* Internal technical knowledge bases

Unlike traditional RAG systems that directly search documents using the **user query**, this project implements **HyDe (Hypothetical Document Embeddings)**.

Instead of embedding the raw query, the system:

1. Generates a **hypothetical answer** to the question.
2. Embeds that hypothetical document.
3. Uses it to retrieve **semantically similar real documents**.

This technique significantly improves retrieval accuracy when dealing with:

* Technical terminology
* Domain-specific language
* Short or ambiguous user queries

The system is built entirely with **open-source models via Hugging Face APIs**, making it suitable for **secure internal deployments** without relying on closed-source AI platforms.

---

# 🚀 Key Features
<img width="1536" height="1024" alt="diagram" src="https://github.com/user-attachments/assets/97c7f7eb-f241-4ea2-a9bd-ff813d04fab0" />

### 🔹 HyDe Retrieval Mechanism

Implements **Hypothetical Document Embeddings (HyDe)** to improve semantic search for technical documentation.

### 🔹 Privacy-First Architecture

Runs on **open-source LLMs** such as:

* `google/flan-t5-large`
* `Mistral-7B`

This ensures **data control and internal security**.

### 🔹 Fast Vector Search

Uses **FAISS (Facebook AI Similarity Search)** for efficient local vector retrieval.

### 🔹 Interactive UI

Built with **Streamlit** to provide:

* PDF upload capability
* Interactive querying
* Visualization of retrieval steps

### 🔹 Transparent Retrieval Logic

Users can view:

* The **AI-generated hypothetical document**
* The **actual retrieved source text**

This helps understand how the system arrives at answers.

---

# 🧠 How It Works (HyDe Logic)

Traditional RAG systems perform:

```
Query  →  Embedding  →  Document Similarity Search
```

HyDe RAG performs:

```
Query → Hypothetical Answer → Embedding → Document Similarity Search
```

### Example Workflow

**User Query**

```
How do I recalibrate the pressure sensor?
```

---

### Step 1 — Generate Hypothesis

The LLM generates a **hypothetical answer**:

> "To recalibrate the pressure sensor, access the maintenance menu, select calibration mode, and apply zero-reference gas..."

This step helps create a **context-rich representation** of the user's intent.

---

### Step 2 — Embed & Retrieve

The hypothetical document is embedded into a vector representation.

That vector is used to search the **real document database** using FAISS.

This retrieves the **most semantically similar sections** from the uploaded PDF.

---

### Step 3 — Generate Final Answer

The retrieved documents are passed back to the LLM to generate a **factually grounded final response**.

```
User Query
   ↓
Generate Hypothetical Answer
   ↓
Embed Hypothesis
   ↓
Vector Search (FAISS)
   ↓
Retrieve Relevant Documents
   ↓
Generate Final Answer
```

---

# 🧰 Tech Stack

| Component         | Technology                               |
| ----------------- | ---------------------------------------- |
| Language          | Python                                   |
| Framework         | LangChain                                |
| Frontend          | Streamlit                                |
| LLM Provider      | Hugging Face Inference API               |
| Generation Models | `google/flan-t5-large`, `Mistral-7B`     |
| Embedding Model   | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database   | FAISS (Facebook AI Similarity Search)    |

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/secure-hyde-rag.git
cd secure-hyde-rag
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Configure Hugging Face API Key

Get a **free API access token** from:

[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Then open `app.py` and replace the placeholder:

```python
hf_api_key = "hf_xxxxxxxxxxxxxxxxxxxxxxxx"
```

---

# ▶️ Usage

## Run the Application

```bash
streamlit run app.py
```

---

## Interact with the System

1️⃣ Open the local URL shown in the terminal (usually)

```
http://localhost:8501
```

2️⃣ Upload a **PDF document**

Examples:

* Technical manuals
* SOP documentation
* Research papers

3️⃣ Wait for the **Processed!** message.

4️⃣ Ask a question about the document.

5️⃣ Expand:

```
View Generated Hypothetical Document
```

to see the **HyDe retrieval process in action.**

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project:

1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

