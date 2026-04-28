import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def generate_response(uploaded_file, openai_api_key, query_text):
    documents = [uploaded_file.read().decode()]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question using only the context below.\n\n{context}"),
            ("human", "{input}"),
        ]
    )
    combine = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine)
    return chain.invoke({"input": query_text})["answer"]


st.set_page_config(page_title="🦜🔗 Ask the Doc App")
st.title("🦜🔗 Ask the Doc App")

uploaded_file = st.file_uploader("Upload an article", type="txt")
query_text = st.text_input(
    "Enter your question:",
    placeholder="Please provide a short summary.",
    disabled=not uploaded_file,
)

result = []
with st.form("myform", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        disabled=not (uploaded_file and query_text),
    )
    submitted = st.form_submit_button(
        "Submit", disabled=not (uploaded_file and query_text)
    )
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner("Calculating..."):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
