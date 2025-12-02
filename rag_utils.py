import os
from typing import Set, List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser


# --- CONFIGURACIÓN GLOBAL (Debe coincidir con la de las otras apps) ---
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
    """Extrae el nombre del archivo y el número de página como fuentes."""
    sources_set = set()
    for doc in docs:
        if 'source' in doc.metadata and 'page' in doc.metadata:
            # Formato: NombreArchivo (Pág X)
            source_name = os.path.basename(doc.metadata['source'])
            sources_set.add(f"{source_name} (Pág {doc.metadata['page'] + 1})")
    return "\n".join(sorted(list(sources_set)))

# ----------------------------------------------------------------------
# CREACIÓN DE LA CADENA DE RAG (Retrieval Augmented Generation)
# ----------------------------------------------------------------------

def create_rag_chain(db: Chroma, llm_model_name: str, theme_filter: Optional[Dict[str, str]] = None):
    """
    Crea la cadena LCEL RAG con un filtro de metadatos opcional.
    
    theme_filter: Diccionario de metadatos para filtrar (ej: {"theme": "historia"}). 
                  Si es None, no se aplica filtro.
    """
    
    # 1. Inicializar el LLM
    llm = OllamaLLM(model=llm_model_name)
    
    # 2. Definir el Prompt
    RAG_PROMPT = PromptTemplate.from_template("""
        Eres un asistente útil especializado en buscar información en los documentos proporcionados.
        Tu tarea es responder a la pregunta del usuario basándote EXCLUSIVAMENTE en el contexto proporcionado.
        Si el contexto no contiene la respuesta, di claramente "No puedo responder basándome en los documentos proporcionados."
        
        CONTEXTO:
        ---
        {context}
        ---
        
        PREGUNTA DEL USUARIO: {question}
        
        Respuesta:
    """)
    
    # 3. Configurar el Retriever (punto de la corrección)
    
    # Creamos un diccionario de argumentos de búsqueda y SOLO incluimos 'filter' si es necesario.
    search_kwargs = {"k": 4} 
    if theme_filter is not None:
        # CORRECCIÓN: Evitamos pasar 'filter': None, lo cual causaba el error 'got None in query'.
        search_kwargs["filter"] = theme_filter

    retriever = db.as_retriever(search_kwargs=search_kwargs)
    
    # 4. Definir la cadena de generación de respuestas RAG
    rag_answer_generation = (
        RAG_PROMPT 
        | llm 
        | StrOutputParser()
    )
    
    # 5. Pipeline LCEL
    
    # 5.1. Preparación de la cadena: Combina la pregunta con la recuperación de documentos.
    # Los documentos recuperados se pasan a 'context' en el prompt.
    rag_chain_prep = RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=RunnablePassthrough(),
        documents=retriever, # Mantiene los documentos originales para extraer fuentes
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
    
    # Cadena simple para Resumen: Input {"text": text_to_summarize} -> LLM -> Output (str)
    summary_chain = (
        SUMMARY_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return summary_chain