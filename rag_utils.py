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
    """Extrae el nombre del archivo y número de página de los documentos recuperados."""
    sources = set()
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'Desconocido'))
        page = doc.metadata.get('page', 'N/A')
        sources.add(f"Archivo: {source}, Página: {page}")
    
    return "\n".join(sorted(list(sources)))


# ----------------------------------------------------------------------
# CREACIÓN DE LA CADENA RAG (Retrieval-Augmented Generation)
# ----------------------------------------------------------------------

def create_rag_chain(db: Chroma, llm_model_name: str, theme_filter: str):
    """
    Crea la cadena RAG usando LCEL, asegurando que los documentos se recuperen
    una sola vez y se pasen tanto para el contexto como para las fuentes.
    """
    
    # 1. Definir el modelo LLM usando OllamaLLM
    try:
        llm = OllamaLLM(model=llm_model_name)
    except ImportError:
        print(f"[ERROR CRÍTICO] Falta la dependencia 'langchain-ollama'. Asegúrate de que esté instalada.")
        return None
    except Exception as e:
        print(f"[ERROR OLLAMA] No se pudo inicializar OllamaLLM con '{llm_model_name}'. Verifique su servicio Ollama. Error: {e}")
        return None
        
    # 2. Definir el Prompt
    RAG_PROMPT = PromptTemplate.from_template("""
        Eres un asistente de recuperación de información experto.
        Utiliza SOLO los siguientes fragmentos de contexto para responder la pregunta del usuario.
        Si la respuesta no se encuentra en el contexto proporcionado, indica claramente que la información no está disponible en los documentos.
        
        CONTEXTO:
        ---
        {context}
        ---
        
        PREGUNTA: {question}
        
        Respuesta Detallada:
    """)
    
    # 3. Configurar el Retriever (siempre búsqueda simple 'similarity' para evitar el error de umbral)
    search_kwargs = {'k': 5} # Recupera los 5 documentos más cercanos
    
    if theme_filter != "TODOS":
        metadata_filter = {"theme": theme_filter}
        search_kwargs['filter'] = metadata_filter

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    # 4. Definición del pipeline LCEL

    # PASO A: Recuperar documentos Y mantener la pregunta
    # Esto es una RunnableParallel que ejecuta el retriever y pasa la pregunta sin modificar.
    # El resultado es un diccionario: {'docs': [Documentos...], 'question': '...'}.
    retrieval_chain = RunnableParallel(
        # 'docs' es una lista de Documentos. El retriever es Runnable.
        docs=retriever, 
        # 'question' es la cadena de texto de entrada que pasa sin cambios
        question=RunnablePassthrough() 
    )

    # PASO B: Generación de respuesta y extracción de fuentes
    # Esto toma el diccionario {'docs': [docs], 'question': 'query'} del PASO A.
    rag_chain_with_sources = retrieval_chain | RunnableParallel(
        # 1. 'answer': Genera la respuesta
        answer=(
            RunnableParallel(
                # Formatea los documentos para el contexto del prompt
                context=RunnableLambda(lambda x: format_docs(x['docs'])), 
                # Pasa la pregunta al prompt
                question=RunnableLambda(lambda x: x['question'])
            )
            | RAG_PROMPT 
            | llm
        ),
        # 2. 'sources': Extrae las fuentes usando los documentos ya recuperados
        sources=RunnableLambda(lambda x: format_sources(x['docs']))
    )
    
    return rag_chain_with_sources

# ----------------------------------------------------------------------
# CREACIÓN DE LA CADENA DE RESUMEN (Summary Chain)
# ----------------------------------------------------------------------

def create_summary_chain(llm_model_name: str):
    """
    Crea una cadena simple para resumir un texto dado.
    """
    # Usamos OllamaLLM para la cadena de resumen
    try:
        llm = OllamaLLM(model=llm_model_name)
    except ImportError:
        print(f"[ERROR CRÍTICO] Falta la dependencia 'langchain-ollama'. Asegúrate de que esté instalada.")
        return None
    except Exception as e:
        print(f"[ERROR OLLAMA] No se pudo inicializar OllamaLLM para Summary. Verifique su servicio Ollama. Error: {e}")
        return None
    
    SUMMARY_PROMPT = PromptTemplate.from_template("""
        Genera un resumen detallado y claro de UNO a DOS párrafos del siguiente texto.
        Solo incluye la información más relevante.

        TEXTO A RESUMIR:
        ---
        {text}
        ---

        Resumen:
    """)
    
    summary_chain = (
        {"text": RunnablePassthrough()} 
        | SUMMARY_PROMPT 
        | llm
    )
    
    return summary_chain