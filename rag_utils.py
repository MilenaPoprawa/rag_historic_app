import os
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, vectorstore_path: str = "chroma_ve"):
        self.vectorstore_path = vectorstore_path
        self.api_key = os.getenv("OPENAI_API_KEY")

        self._load_vectorstore()
        self._setup_llm()
        self._setup_retriever_and_pipeline()

    def _load_vectorstore(self):
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.api_key,
        )
        self.vectorstore = Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=self.embedding_model,
        )

    def _setup_llm(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.2,
            max_tokens=2000,
            api_key=self.api_key,
            verbose=False,
        )

    def _setup_retriever_and_pipeline(self):
        prompt_template = """
Eres un asistente de inteligencia artificial especializado en historia. Respondes preguntas únicamente utilizando el contexto documental proporcionado.

Reglas:
- No inventes información. Si el contexto no tiene la respuesta, dilo claramente.
- Sé específico y fundamenta tus respuestas citando fragmentos relevantes del contexto (usa comillas si es posible).
- Resume con claridad y precisión. Evita redundancias.
- No incluyas información externa al contexto.

Contexto:
{context}

Pregunta: {question}

Respuesta fundamentada únicamente en el contexto anterior:
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        self.rag_pipeline = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def query(self, question: str):
        start_time = time.time()
        result = self.rag_pipeline.invoke({"query": question})
        end_time = time.time()
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        compute_time = end_time - start_time
        return answer, source_docs, compute_time
