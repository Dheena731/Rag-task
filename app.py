import streamlit as st
import os
import tempfile

from build_faiss import build_faiss_index_from_pdf
from query_resume import query_index, build_context_from_chunks
from llm import ask_profile_bot

st.title("Resume RAG Query Application")

st.markdown("""
Upload a resume PDF, build the FAISS index, and ask questions about the resume.
The system will retrieve relevant sections and generate answers using an LLM.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

index_built = st.session_state.get("index_built", False)

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_pdf_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # Button to build index
    if st.button("Build FAISS Index"):
        with st.spinner("Building index... This may take a few seconds."):
            try:
                build_faiss_index_from_pdf(temp_pdf_path)
                st.session_state.index_built = True
                st.success("Index built successfully!")
            except Exception as e:
                st.error(f"Error building index: {str(e)}")

# Question input
question = st.text_input("Ask a question about the resume:")

# Query button
if st.button("Get Answer"):
    if not st.session_state.get("index_built", False):
        st.error("Please upload a PDF and build the index first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                # Retrieve relevant chunks
                chunks = query_index(question)
                if not chunks:
                    st.warning("No relevant chunks found in the resume.")
                else:
                    # Build context
                    context = build_context_from_chunks(chunks)
                    # Generate answer
                    answer = ask_profile_bot(question, context)
                    st.subheader("Answer:")
                    st.write(answer)
                    # Optionally show retrieved chunks
                    with st.expander("View Retrieved Context"):
                        st.text_area("Context", context, height=200)
            except Exception as e:
                st.error(f"Error during query: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, FAISS, Sentence Transformers, and Groq LLM.")