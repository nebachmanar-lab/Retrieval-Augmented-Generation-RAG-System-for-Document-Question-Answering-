import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import ollama
import streamlit as st

# Cache pour le modèle d'embedding (chargé une seule fois)
@st.cache_resource
def load_embedding_model(model_name="thenlper/gte-small", device="cpu"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )

#Initialise la configuration du pipeline
class RAGPipeline:
    def __init__(self, persist_directory="./chroma_app", embedding_model_name="thenlper/gte-small", use_pinecone=False):
        self.persist_directory = persist_directory
        self.use_pinecone = use_pinecone
        self.embedding_model = load_embedding_model(embedding_model_name, "cpu")
        self.vectorstore = None
        self.retriever = None
        if use_pinecone:
            self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            self.index_name = "rag-index-pinecone"

#Indexation
    def index_documents(self, file_paths):
        """Charge, découpe et indexe une liste de fichiers PDF."""
        all_docs = []
        for file_path in file_paths:
            # Étape 1 : Charger les PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        # Étape 2 : Découper les documents en chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=16,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)

        if self.use_pinecone:
            existing = [idx["name"] for idx in self.pc.list_indexes()]
            if self.index_name not in existing:
                self.pc.create_index(name=self.index_name, dimension=384, metric="cosine")
            self.vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                index_name=self.index_name
            )
        else:
            # Étape 3 : Créer les embeddings et stocker
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
                collection_name="rag_collection"
            )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def delete_all_documents(self):
        """Supprime tous les documents (réinitialise la base)."""
        if self.vectorstore is not None:
            if self.use_pinecone:
                self.pc.delete_index(self.index_name)
                self.pc.create_index(name=self.index_name, dimension=384, metric="cosine")
            else:
                self.vectorstore.delete_collection()
        self.vectorstore = None
        self.retriever = None

    def query(self, question, model="gemma3"):
        """Pose une question, renvoie la réponse et le contexte."""
        if self.retriever is None:
            raise ValueError("Aucun document indexé. Veuillez d'abord télécharger des fichiers.")
        # Étape 1 : Recherche sémantique
        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        # Étape 2 : Génération de la réponse avec Ollama
        prompt = f"""Utilise le contexte ci-dessous pour répondre à la question.
Si tu ne connais pas la réponse, dis simplement "Je ne sais pas".

Contexte :
{context}

Question : {question}
Réponse :"""
    # Étape 3 : Appel au LLM

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "answer": response["message"]["content"],
            "context": context,
            "docs": docs
        }