# Ask the Doc App

A Streamlit app that answers questions about an uploaded text document using LangChain, OpenAI embeddings, and a Chroma vector store.

Upload a `.txt` file, type a question, paste your OpenAI API key, click **Submit**, and the app returns an answer grounded in the document.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and click **New app**.
3. Pick the repo, branch `main`, main file `app.py`, then **Deploy**.

## How it works

`app.py` splits the uploaded document into chunks, embeds them with `OpenAIEmbeddings`, indexes them in an in-memory FAISS vector store, and runs a retrieval chain (`create_retrieval_chain` + `create_stuff_documents_chain`) backed by `gpt-4o-mini` to answer the question.
