import streamlit as st
import pdfplumber
import requests
import numpy as np
import faiss
import html
st.set_page_config(page_title="RAG App", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧠📄 Retrieval-Augmented Generation (RAG) App</h1>", unsafe_allow_html=True)
with st.sidebar:
    st.header("Configuration")

    HF_TOKEN = st.text_input("🔑 Hugging Face API Token", type="password",
                             help="Enter your personal Hugging Face access token.")

    if not HF_TOKEN:
        st.warning("⚠️ Please enter your Hugging Face API token to continue.")
        st.stop()



# ==== Hugging Face Chat Generation ====
def hf_chat_generate(model: str, prompt: str, options: dict, context: str = None) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    full_prompt = f"{context}\n\nQuestion: {prompt}" if context else prompt
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "temperature": options.get("temperature", 0.7),
            "max_new_tokens": options.get("max_length", 256),
            "top_p": options.get("top_p", 0.9),
        }
    }

    endpoint = f"https://api-inference.huggingface.co/models/{model}"
    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        generated = ""
        if isinstance(result, list):
            generated = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            generated = result.get("generated_text", "")

        # Remove repeated prompt from output
        if context:
            combined_prompt = f"{context}\n\nQuestion: {prompt}"
            if generated.startswith(combined_prompt):
                generated = generated[len(combined_prompt):].strip()

        # Clean up unwanted lines and format
        lines = generated.strip().split("\n")
        lines = [line for line in lines if "and what you will learn" not in line.lower()]
        cleaned = "\n".join(lines).strip()

        # Ensure response starts with "Answer:"
        if not cleaned.lower().startswith("answer:"):
            cleaned = "Answer: " + cleaned.lstrip(": ")

        return cleaned or "Answer: (No meaningful output)"
    return "❌ Error: " + response.text

# ==== Hugging Face Embeddings ====
def hf_get_embedding(model: str, text: str) -> list:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    endpoint = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    response = requests.post(endpoint, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and isinstance(data[0], list):
            return np.mean(data, axis=0).tolist()
        return data
    else:
        st.warning(f"Embedding error: {response.status_code} - {response.text}")
        return []

# ==== Extract PDF Text ====
def extract_text_and_images(pdf_file):
    text, images = "", []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            images.extend(page.images)
    return text, images

# ==== Split Text into Chunks ====
def split_text(text, max_tokens=300):
    words = text.split()
    chunks, current_chunk = [], []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ==== Sidebar Config ====
with st.sidebar:
    st.header("Configuration")
    uploaded_pdf = st.file_uploader("📄 Upload a PDF", type="pdf")

    available_models = [
        "mistralai/Mistral-7B-Instruct-v0.3",  # 7B parameters, ~13GB
        "HuggingFaceH4/zephyr-7b-beta",
        "tiiuae/falcon-7b-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
    ]

    selected_model = st.selectbox("🧠 Select LLM", available_models)

    embedding_model = st.selectbox("🔎 Embedding Model", [
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/multilingual-e5-base"
    ])

    temperature = st.slider("🎯 Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("🔢 Max New Tokens", 64, 1024, 256)

    user_query = st.text_input("💬 Ask a question:")
    ask_button = st.button("Ask")

# ==== Initialize FAISS ====
faiss_index = None
all_chunks = []

# ==== PDF Upload & Processing ====
if uploaded_pdf:
    with st.spinner("🔍 Processing PDF..."):
        text, images = extract_text_and_images(uploaded_pdf)
        chunks = split_text(text)
        st.success(f"✅ Extracted {len(chunks)} chunks from PDF.")

        embeddings = []
        for chunk in chunks:
            emb = hf_get_embedding(embedding_model, chunk)
            if emb:
                embeddings.append(emb)
                all_chunks.append(chunk)

        if embeddings:
            dim = len(embeddings[0])
            faiss_index = faiss.IndexFlatL2(dim)
            faiss_index.add(np.array(embeddings).astype("float32"))

    # ==== Handle Question ====
    if ask_button and user_query:
        with st.spinner("🔎 Searching relevant chunks..."):
            query_emb = hf_get_embedding(embedding_model, user_query)
            if query_emb and faiss_index:
                D, I = faiss_index.search(np.array([query_emb]).astype("float32"), k=5)
                relevant_chunks = [all_chunks[i] for i in I[0]]

                st.subheader("📚 Retrieved Context Chunks")
                for idx, chunk in enumerate(relevant_chunks, 1):
                    with st.expander(f"Chunk {idx}"):
                        st.markdown(f"<div style='text-align: justify;'>{html.escape(chunk)}</div>", unsafe_allow_html=True)

                # ==== Generate Response ====
                with st.spinner("🤖 Generating answer..."):
                    context = "\n\n".join(relevant_chunks)
                    response = hf_chat_generate(
                        model=selected_model,
                        prompt=user_query,
                        options={"temperature": temperature, "max_length": max_tokens, "top_p": 0.9},
                        context=context
                    )

                # ==== Display Response ====
                st.subheader("🤖 Model Response")
                st.markdown(f"""
                    <style>
                        .model-response {{
                            padding: 1rem;
                            border-left: 5px solid #4CAF50;
                            border-radius: 4px;
                            font-size: 1rem;
                            line-height: 1.6;
                        }}
                        @media (prefers-color-scheme: dark) {{
                            .model-response {{
                                background-color: #1e1e1e;
                                color: #f0f0f0;
                            }}
                        }}
                        @media (prefers-color-scheme: light) {{
                            .model-response {{
                                background-color: #f9f9f9;
                                color: #000000;
                            }}
                        }}
                    </style>
                    <div class="model-response">
                        {response.replace('\n', '<br>')}
                    </div>
                """, unsafe_allow_html=True)

else:
    st.info("📥 Upload a PDF from the sidebar to begin.")