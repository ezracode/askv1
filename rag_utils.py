import os
from typing import Set, List
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURACIÓN GLOBAL ---
LLM_MODEL_NAME = "mistral"
EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# ----------------------------------------------------------------------
# FUNCIÓN DE INICIALIZACIÓN DE EMBEDDINGS (Centralizada aquí)
# ----------------------------------------------------------------------

def initialize_embeddings() -> HuggingFaceEmbeddings:
    """Inicializa el modelo de embeddings."""
    # Nota: Este modelo es multilingüe y eficiente para RAG.
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )

# ----------------------------------------------------------------------
# FUNCIONES DE AYUDA PARA EL PIPELINE LCEL
# ----------------------------------------------------------------------

def format_docs(docs: List[Document]) -> str:
    """Formatea los documentos recuperados en una sola cadena de texto para el Prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def format_sources(docs: List[Document]) -> str:
    """Extrae el nombre del archivo y la página de los metadatos para el usuario."""
    sources = set()
    for doc in docs:
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page', 'N/A')
        theme = doc.metadata.get('theme', 'N/A')
        file_name = os.path.basename(source) 
        sources.add(f"[Tema: {theme}] Archivo: {file_name} (Página: {page})")
    return "\n".join(sources)

# ----------------------------------------------------------------------
# CREACIÓN DE LA CADENA RAG (LCEL CORREGIDO)
# ----------------------------------------------------------------------

def create_rag_chain(db: Chroma, llm_model_name: str, selected_theme: str):
    """
    Crea la cadena RAG (Retrieval Augmented Generation) usando LangChain Expression Language (LCEL).

    Parámetros:
        db (Chroma): La base de datos vectorial con los documentos.
        llm_model_name (str): Nombre del modelo LLM a usar (ej: 'mistral').
        selected_theme (str): El tema seleccionado para filtrar la búsqueda (o 'TODOS').
    """
    
    # 1. Inicializar el LLM
    llm = OllamaLLM(model=llm_model_name)

    # 2. Definir el Prompt
    RAG_PROMPT = PromptTemplate.from_template("""Usa solo el siguiente contexto para responder a la pregunta. 
Si la respuesta no está contenida en el contexto, simplemente di que no tienes suficiente información.
El contexto proviene del tema: {theme}

{context}

Pregunta: {question}
Respuesta Concisa:""")

    # 3. Configurar el Filtro y el Retriever
    # Inicializamos los argumentos de búsqueda solo con 'k'
    search_kwargs = {"k": 5}
    
    if selected_theme and selected_theme.upper() != 'TODOS':
        # Si se selecciona un tema específico, añadimos el filtro a los argumentos.
        search_filter = {"theme": selected_theme}
        search_kwargs["filter"] = search_filter # AÑADIMOS 'filter' SOLO SI NO ES 'TODOS'

    # Configurar el Retriever con los argumentos de búsqueda condicionales
    retriever = db.as_retriever(
        search_kwargs=search_kwargs
    )

    # 4. Pipeline para generar la Respuesta (Core): 
    # Recibe {'documents': docs, 'question': q} y devuelve la respuesta del LLM.
    rag_answer_generation = (
        {
            "context": RunnableLambda(lambda x: format_docs(x["documents"])), 
            "question": RunnableLambda(lambda x: x["question"]),
            "theme": RunnableLambda(lambda x: selected_theme) # Añadimos el tema al contexto del prompt
        }
        | RAG_PROMPT
        | llm
    )
    
    # 5. Pipeline Final con Fuentes (LCEL CORREGIDO - Uso de RunnableParallel)
    
    # 5.1. Preparación de la cadena: Combina la pregunta con la recuperación de documentos.
    rag_chain_prep = RunnableParallel(
        documents=retriever,
        question=RunnablePassthrough(),
    )
    
    # 5.2. Cadena completa: Encadena la preparación con la ejecución paralela
    # para obtener la 'answer' y las 'sources' al mismo tiempo.
    rag_chain_with_sources = (
        rag_chain_prep
        | RunnableParallel(
            answer=rag_answer_generation, 
            sources=RunnableLambda(lambda x: format_sources(x["documents"]))
        )
    )
    
    return rag_chain_with_sources

# ----------------------------------------------------------------------
# CREACIÓN DE LA CADENA DE RESUMEN (Summary Chain)
# ----------------------------------------------------------------------

def create_summary_chain(llm_model_name: str):
    """
    Crea una cadena simple para resumir un texto dado.
    """
    llm = OllamaLLM(model=llm_model_name)
    
    SUMMARY_PROMPT = PromptTemplate.from_template("""
        Genera un resumen detallado y claro de UNO a DOS párrafos del siguiente texto.
        Solo incluye la información más relevante.

        TEXTO A RESUMIR:
        ---
        {text}
        ---

        Resumen:
    """)
    
    # La cadena solo necesita el texto a resumir
    summary_chain = (
        {"text": RunnablePassthrough()}
        | SUMMARY_PROMPT
        | llm
    )
    return summary_chain