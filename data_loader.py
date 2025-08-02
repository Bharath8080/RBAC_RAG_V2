from fastapi.security import api_key
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()


DATA_DIR = "data"
FAISS_DIR = "vector_store"
departments = ["finance", "hr", "marketing", "engineering", "general"]


embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding = OpenAIEmbeddings(api_key= os.getenv("OPENAI_API_KEY"))
# embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",api_key=os.getenv("GOOGLE_API_KEY"))



def load_and_store():
    for dept in departments:
        path = os.path.join(DATA_DIR, dept)
        if not os.path.exists(path):
            print(f"Skipping missing folder: {dept}")
            continue

        print(f"Processing {dept}...")
        all_chunks = []

        loaders = [
            DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader),
            DirectoryLoader(path, glob="*.md", loader_cls=UnstructuredMarkdownLoader),
            DirectoryLoader(path, glob="*.csv", loader_cls=CSVLoader),
        ]

        for loader in loaders:
            docs = loader.load()
            for doc in docs:
                doc.metadata["allowed_roles"] = [dept.capitalize()]
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)    
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

        db = FAISS.from_documents(documents=all_chunks, embedding=embedding)
        db.save_local(folder_path=os.path.join(FAISS_DIR, dept)) 
        print(f"Indexed {dept} with {len(all_chunks)} chunks.")
if __name__=="__main__":
    load_and_store()        

