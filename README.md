# 🚀 Neura-IQ Multimodal AI Research Assistant

**Neura-IQ Multimodal AI Research Assistant** is a locally-running multimodal AI system that lets you ingest PDF documents (text, tables, and images), chat with them using RAG, and fine-tune a lightweight LLM on your own data — all with zero cloud dependencies.

---

## ✨ Features

- **📚 Multimodal PDF Ingestion** — Extracts text, tables, and images from PDFs using `unstructured` (HI_RES strategy)
- **🔍 Multimodal RAG** — Retrieves text, table, and image chunks from ChromaDB and answers questions using LLaVA 13B
- **🤖 General LLM Chat** — Free-form chat with LLaVA 13B, supports image uploads
- **🧠 Fine-Tuning** — Auto-generates Q&A pairs from your indexed documents and fine-tunes **FLAN-T5-small** using **LoRA (PEFT)** — fully on CPU
- **🔐 Auth System** — Login page, role-based access (admin/user), session timeout, brute-force lockout, user management panel
- **🗃️ Fully Local** — Ollama (LLaVA + Nomic embeddings), ChromaDB, SQLite — no API keys, no cloud

---

## 🏗️ Architecture

```
User (Browser)
    │
    ▼
Streamlit App (core.py)
    │
    ├─── Auth Layer (auth_ui.py + auth_utils.py)
    │         └── SQLite (users.db) ← bcrypt + session timeout
    │
    └─── EnhancedMultimodalRAG
              ├── Ollama (local)
              │     ├── llava:13b        ← generation + image analysis
              │     └── nomic-embed-text ← embeddings
              ├── ChromaDB (persistent)
              │     └── "multimodal_rag" collection
              ├── unstructured.io
              │     └── PDF → text + tables + images
              └── HuggingFace (CPU)
                    ├── FLAN-T5-small (base)
                    └── LoRA fine-tuned T5
```

---

## 📋 Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Required Ollama models:
  ```bash
  ollama pull llava:13b
  ollama pull nomic-embed-text
  ```
- System dependencies for `unstructured` PDF extraction:
  ```bash
  sudo apt install tesseract-ocr poppler-utils
  ```

---

## 🚀 Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Torque4708/Neura-IQ.git
cd Neura-IQ

# 2. Create and activate a virtual environment
python -m venv env
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running
ollama serve

# 5. Launch the app
streamlit run core.py --server.port 8501
```

Or use the boot script (update paths first):
```bash
bash neura_iq_boot.sh
```

---

## 🔐 Default Login

| Username | Password | Role  |
|----------|----------|-------|
| `anand`  | `anand123` | Admin |

> Change the default password in `auth_utils.py` before sharing or deploying.

---

## 📂 Data Storage

| Path | Contents |
|---|---|
| `data/pdfs/` | Uploaded PDF files |
| `data/figures/<pdf_name>/` | Extracted images per PDF |
| `data/tables/<pdf_name>/` | Extracted tables as CSVs |
| `data/chromadb/` | ChromaDB vector store |
| `data/metadata.json` | PDF processing metadata |
| `data/finetuned_t5_lora/` | Saved fine-tuned model |
| `data/qa_dataset.json` | Generated Q&A pairs |
| `users.db` | SQLite user database |

---

## 🧠 Fine-Tuning Pipeline

1. **Index your PDFs** via the *Create New Index* mode
2. Go to **Fine-Tune Model** → *Generate Q&A* tab — LLaVA auto-generates question-answer pairs from your chunks
3. Switch to *Fine-Tune* tab — trains FLAN-T5-small with LoRA on CPU
4. In the *Test Model* tab — compare **Base T5** vs **Fine-tuned T5** side-by-side

**LoRA Config:** rank=16, alpha=32, targets all attention and FFN projections (`q, k, v, o, wi_0, wi_1, wo`)

---

## 🗂️ Project Structure

```
Neura-IQ/
├── core.py            # Main app — RAG engine, fine-tuning, all UI modes
├── auth_utils.py      # Auth backend — SQLite, bcrypt, session management
├── auth_ui.py         # Auth frontend — login page, user management UI
├── requirements.txt   # Python dependencies
├── neura_iq_boot.sh   # Shell launcher (update paths before use)
└── data/              # Auto-created at runtime
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM + Vision | LLaVA 13B via Ollama |
| Embeddings | Nomic Embed Text via Ollama |
| Vector Store | ChromaDB |
| PDF Parsing | unstructured (HI_RES) |
| Fine-tuning | HuggingFace Transformers + PEFT/LoRA |
| Base model | google/flan-t5-small |
| Auth DB | SQLite + bcrypt (passlib) |

---

## 📄 License

MIT License — feel free to use, modify, and build on it.
