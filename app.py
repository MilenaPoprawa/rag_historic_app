import streamlit as st
from rag_utils import RAGSystem
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Histórico España → Latinoamérica", layout="wide")

@st.cache_resource(show_spinner=True)
def load_rag():
    # Just load the prebuilt vectorstore from persistent directory
    vectorstore_path = "chroma_ve"  # Folder you upload with all vectorstore data
    return RAGSystem(vectorstore_path=vectorstore_path)

rag = load_rag()

st.title("📚 RAG Histórico: Migración España a Latinoamérica")

st.markdown(
    """
    Este sistema responde preguntas históricas usando documentos PDF relevantes sobre migración de España a Latinoamérica.
    """
)

query = st.text_input("Escribe tu pregunta aquí:")

if query:
    with st.spinner("Buscando respuesta..."):
        answer, source_docs, elapsed = rag.query(query)

    st.markdown("### Respuesta:")
    st.write(answer)

    st.markdown(f"*(Tiempo de consulta: {elapsed:.2f} segundos)*")

    if source_docs:
        st.markdown("### Documentos Fuente:")
        for i, doc in enumerate(source_docs):
            title = doc.metadata.get("title", "Sin título")
            content_preview = doc.page_content[:300].replace("\n", " ") + "..."
            st.markdown(f"**Documento {i+1}:** {title}")
            st.markdown(f"> {content_preview}")
            st.markdown("---")