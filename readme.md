# 📚 RAG Histórico: Migración de España a Latinoamérica

Este proyecto es una aplicación web construida con Streamlit que utiliza un sistema RAG (Retrieval-Augmented Generation) para responder preguntas sobre la migración histórica de España hacia América Latina.  
Usa documentos PDF procesados y almacenados en una base vectorial (`Chroma`) para ofrecer respuestas fundamentadas con contexto documental.

---

## 🧠 ¿Cómo funciona?

El sistema:
1. **Carga una base de datos vectorial preprocesada (`chroma_ve`)** con fragmentos de libros históricos.
2. **Recibe preguntas del usuario**.
3. **Recupera los documentos más relevantes**.
4. **Genera respuestas usando un modelo LLM (OpenAI GPT-4)** basadas únicamente en el contexto recuperado.
