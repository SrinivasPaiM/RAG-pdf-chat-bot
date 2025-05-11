# 🧠📄 Retrieval-Augmented Generation (RAG) App

A **Streamlit-based RAG (Retrieval-Augmented Generation)** application that allows you to upload a PDF, extract and embed its contents, and ask questions powered by Hugging Face LLMs. It retrieves relevant context chunks using FAISS and returns AI-generated answers grounded in your uploaded document.

---

## 🚀 Features

- 📄 PDF Upload & Text Extraction using `pdfplumber`
- 🔍 Semantic search over document chunks via **FAISS**
- 💬 Question Answering using Hugging Face **LLMs**
- 🔐 Hugging Face **inference API** (with token)
- 🌍 Supports multilingual embedding models
- 📊 UI built using **Streamlit**, with chunk display & styled output

---

## Screanshots

![Screenshot 2025-04-05 170705](https://github.com/user-attachments/assets/cd1a9f7d-bd35-4b13-ab30-e8b461d7ea88)

---


## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **LLM API**: Hugging Face Inference API
- **Embedding Models**: `sentence-transformers`, `multilingual-e5`
- **Vector Store**: FAISS
- **PDF Parsing**: `pdfplumber`

---

## 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/rag-app.git
cd rag-app

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 📝 Requirements (`requirements.txt`)
```txt
streamlit
pdfplumber
requests
faiss-cpu
numpy
```

---

## 🔐 Hugging Face API Token

1. Create an account on [Hugging Face](https://huggingface.co)
2. Go to [Access Tokens](https://huggingface.co/settings/tokens)
3. Generate a **Read token**
4. Replace the token in the script:

```python
HF_TOKEN = "your_huggingface_token"
```

**⚠️ Do NOT commit private tokens to public repositories!**

---

## ▶️ Usage

```bash
streamlit run app.py
```

1. Upload a PDF from the sidebar.
2. Select the LLM and embedding model.
3. Ask a question in the input box.
4. View retrieved chunks and generated response.

---

## 🧠 Available LLMs

- `mistralai/Mistral-7B-Instruct-v0.3`
- `HuggingFaceH4/zephyr-7b-beta`
- `tiiuae/falcon-7b-instruct`
- `microsoft/Phi-3-mini-4k-instruct`

---

## 🔎 Embedding Models

- `sentence-transformers/all-MiniLM-L6-v2` (fast and accurate)
- `intfloat/multilingual-e5-base` (better for multilingual PDFs)

---

## 📂 Folder Structure

```bash
.
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 📄 License

This project is open-source under the MIT License.
