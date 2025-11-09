import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import joblib

def build_rag_pipeline(crop_info_path="data/crop_info", faiss_index_path="models/faiss_index.pkl"):
    """
    Builds and saves the FAISS vector store for crop information.
    """
    documents = []
    for filename in os.listdir(crop_info_path):
        if filename.endswith(".md"):
            file_path = os.path.join(crop_info_path, filename)
            loader = UnstructuredMarkdownLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        print(f"No markdown documents found in {crop_info_path}. Please check the path and file types.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    # Using a common sentence transformer model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector store
    db = FAISS.from_documents(texts, embeddings)

    # Save the FAISS index
    joblib.dump(db, faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")

if __name__ == "__main__":
    build_rag_pipeline()
