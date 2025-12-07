import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- API Configuration ---
# REPLACE THIS WITH YOUR ACTUAL API KEY
YOUR_API_KEY_HERE_key = "YOUR_API_KEY_HERE"

# --- Page Configuration ---
st.set_page_config(
    page_title="HyDe RAG Explorer",
    page_icon="🔍",
    layout="wide"
)

# --- Header & CSS ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #3366ff;}
    .sub-header {font-size: 1.2rem; color: #555;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .chat-message {padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;}
    .chat-message.user {background-color: #f0f2f6;}
    .chat-message.bot {background-color: #e6f3ff;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔍 HyDe: Hypothetical Document Embeddings</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Retrieval-Augmented Generation using Hugging Face APIs</div>', unsafe_allow_html=True)

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("**Model Settings**")
    # Switched to google/flan-t5-large. 
    # This model is extremely reliable on the free HF API and avoids the "conversational" task error.
    repo_id = st.text_input("Model Repo ID", value="google/flan-t5-large")
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, help="Lower is more factual")
    
    st.divider()
    
    st.info("""
    **What is HyDe?**
    Instead of searching for your question directly, HyDe:
    1. Generates a *hypothetical* answer.
    2. Embeds that hypothetical answer.
    3. Searches for real documents similar to the hypothesis.
    """)

# --- Helper Functions ---

@st.cache_resource
def get_embeddings():
    # We use local embeddings (CPU/GPU) to ensure stability and avoid API rate limits on massive indexing
    # all-MiniLM-L6-v2 is fast and efficient for this purpose
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    """Loads and splits a PDF file into chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    # Split text into chunks
    # CRITICAL CHANGE: Reduced chunk_size to 300 because Flan-T5 has a small context window (512 tokens).
    # If we feed it too much text, it will crash or truncate the input.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    # Clean up temp file
    os.remove(tmp_path)
    return splits

@st.cache_resource
def create_vector_store(_splits):
    """Creates a FAISS vector store from document splits."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(_splits, embeddings)
    return vectorstore

def get_llm(api_key, repo_id, temperature):
    """Initialize the Hugging Face Inference Endpoint."""
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=api_key,
        temperature=temperature,
        max_new_tokens=512  # Fixed: Use max_new_tokens instead of max_length
    )

# --- HyDe Implementation ---

def generate_hypothetical_document(query, llm):
    """
    Step 1 of HyDe: Generate a hypothetical answer to the query.
    """
    hyde_template = """Please write a short, plausible scientific passage that answers the question.
    Question: {question}
    Passage:"""
    
    prompt = PromptTemplate.from_template(hyde_template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": query})

def retrieve_with_hyde(query, hypothetical_doc, vectorstore, k=4):
    """
    Step 2 of HyDe: Use the hypothetical document to retrieve real chunks.
    """
    # Embed the hypothetical document and search
    # Note: We search using the hypothetical doc, not the raw query
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(hypothetical_doc)

def generate_final_answer(query, retrieved_docs, llm):
    """
    Step 3: Generate the final answer using retrieved real documents.
    """
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    rag_template = """Answer the question based ONLY on the context below.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate.from_template(rag_template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})

# --- Main Application Logic ---

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_file is not None:
    # Check if API key is set
    if YOUR_API_KEY_HERE_key == "YOUR_HUGGINGFACE_API_KEY_HERE" or not YOUR_API_KEY_HERE_key:
        st.error("⚠️ Please set your Hugging Face API Token in the code (variable `YOUR_API_KEY_HERE_key`).")
    else:
        with st.spinner("Processing PDF... (Parsing & Indexing)"):
            try:
                # Process PDF and Create Vector Store
                # We use session state to avoid re-processing on every interaction
                if "vectorstore" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
                    splits = process_pdf(uploaded_file)
                    st.session_state.vectorstore = create_vector_store(splits)
                    st.session_state.current_file = uploaded_file.name
                    st.success(f"Processed {len(splits)} chunks!")
                
                vectorstore = st.session_state.vectorstore
                
                # Query Interface
                query = st.text_input("Ask a question about your document:")
                
                if query:
                    llm = get_llm(YOUR_API_KEY_HERE_key, repo_id, temperature)
                    
                    # --- HyDe Process Visualization ---
                    
                    # 1. Generate Hypothesis
                    with st.status("Running HyDe Pipeline...", expanded=True) as status:
                        st.write("1️⃣ Generating Hypothetical Document...")
                        hypothetical_doc = generate_hypothetical_document(query, llm)
                        st.caption(f"**Hypothesis Preview:** {hypothetical_doc[:200]}...")
                        
                        # 2. Retrieve
                        st.write("2️⃣ Retrieving Real Documents using Hypothesis...")
                        retrieved_docs = retrieve_with_hyde(query, hypothetical_doc, vectorstore)
                        st.caption(f"Found {len(retrieved_docs)} relevant chunks.")
                        
                        # 3. Final Answer
                        st.write("3️⃣ Synthesizing Final Answer...")
                        answer = generate_final_answer(query, retrieved_docs, llm)
                        status.update(label="HyDe Pipeline Complete!", state="complete", expanded=False)
                    
                    # --- Display Results ---
                    st.markdown("### 💡 Answer")
                    st.markdown(f">{answer}")
                    
                    st.divider()
                    
                    # Show Sources
                    with st.expander("📚 View Retrieved Context (Sources)"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Source {i+1}**")
                            st.markdown(doc.page_content)
                            st.divider()
                            
                    # Show full Hypothesis
                    with st.expander("👻 View Generated Hypothetical Document"):
                        st.info("This text was halluncinated by the AI to help find the right content, but was NOT used in the final answer.")
                        st.write(hypothetical_doc)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")