import streamlit as st
from rag_pipeline import RAGPipeline
import tempfile
import os
import json

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("🤖 RAG Assistant - Industrialisation")

# Initialisation unique du pipeline
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(use_pinecone=False)

pipeline = st.session_state.pipeline

# --- Barre latérale ---
with st.sidebar:
    st.header("Gestion des documents")
    uploaded_files = st.file_uploader(
        "Téléchargez des fichiers PDF",
        type=["pdf"],
        accept_multiple_files=True
    )
    if uploaded_files:
        file_paths = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                file_paths.append(tmp.name)
        with st.spinner("Indexation en cours..."):
            pipeline.index_documents(file_paths)
        st.success(f"{len(file_paths)} document(s) indexé(s)")

    if st.button("Supprimer tous les documents"):
        pipeline.delete_all_documents()
        st.success("Base vidée")

    st.header("Options avancées")
    use_pinecone = st.checkbox("Utiliser Pinecone (base cloud)", value=pipeline.use_pinecone)
    if use_pinecone != pipeline.use_pinecone:
        st.session_state.pipeline = RAGPipeline(use_pinecone=use_pinecone)
        st.rerun()   # utilisez st.rerun() pour Streamlit ≥1.27, sinon st.experimental_rerun()

    st.header("Modèle LLM")
    available_models = ["gemma3", "llama3", "mistral"]
    model_name = st.selectbox("Choisissez un modèle", available_models)

# --- Zone principale ---
st.header("Posez votre question")
question = st.text_input("Question :", placeholder="Exemple : Quels sont les avantages de la robotique ?")

if question:
    with st.spinner("Recherche et génération..."):
        try:
            result = pipeline.query(question, model=model_name)
            st.subheader("Réponse")
            st.write(result["answer"])
            with st.expander("Contexte utilisé"):
                st.write(result["context"])
            with st.expander("Sources (chunks)"):
                for i, doc in enumerate(result["docs"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
        except Exception as e:
            st.error(f"Erreur : {e}")

# --- Option : Tester avec questionSource.json ---
with st.sidebar:
    st.header("Évaluation automatique")
    if st.button("Tester avec questionSource.json"):
        try:
            with open("questionSource.json", "r", encoding="utf-8") as f:
                dataset = json.load(f)
            questions = []
            for doc in dataset:
                for qa in doc["qa_pairs"]:
                    questions.append(qa["question"])
            results = []
            progress_bar = st.progress(0)
            for i, q in enumerate(questions):
                r = pipeline.query(q, model=model_name)
                results.append({"question": q, "answer": r["answer"]})
                progress_bar.progress((i+1)/len(questions))
            with open("questionResults_app.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            st.success("Évaluation terminée. Résultats sauvegardés dans questionResults_app.json")
        except Exception as e:
            st.error(f"Erreur : {e}")