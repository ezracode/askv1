import streamlit as st
import os
import json
from typing import Set, List, Optional
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate 
# Importamos las herramientas de LCEL
from langchain_core.runnables import RunnablePassthrough, RunnableLambda 
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM 

# Importaciones de constantes de ruta desde el script de indexación
from index_data import (
    DATA_PATH, 
    EMBED_MODEL_NAME, 
    CHROMA_DB_PATH_RECURSIVE, 
    CHROMA_DB_PATH_FIXED, 
    CHROMA_DB_PATH_STRUCTURED,
    initialize_embeddings
)

# --- CONFIGURACIÓN CENTRAL ---
TTS_MODEL_NAME = "gemini-2.5-flash-preview-tts"
TTS_VOICE_NAME = "Charon" 
TTS_TEMPLATE_FILE = "tts_component.html" 
# LLM Local de Ollama
LLM_MODEL_NAME = "mistral" 
PROCESSED_LOG = "./processed_files.txt"
API_KEY_GLOBAL = os.environ.get('__api_key', '')

# --- UTILIDADES DE CHAIN DE LANCHAIN ---

# Función auxiliar para formatear el contexto (Documentos) y agregar fuentes
def format_docs(docs):
    text_content = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return text_content

def create_rag_chain(llm_model_name: str, db: Chroma, metadata_filter: dict):
    """Crea y retorna la cadena RAG con filtro de metadatos."""
    
    # 1. Configuración del Retriever con filtro de metadatos
    retriever = db.as_retriever(
        search_kwargs={"filter": metadata_filter, "k": 6} # 6 chunks
    )
    
    # 2. Plantilla del Prompt
    template_str = """
    Eres un asistente de recuperación de información experto.
    Usa SÓLO el siguiente contexto para responder a la pregunta.
    Si la respuesta no se encuentra en el contexto proporcionado, responde honestamente que no tienes la información disponible.

    Contexto: {context}
    Pregunta: {question}

    Respuesta con fuentes:
    """
    prompt_template = PromptTemplate.from_template(template_str)
    
    # 3. Inicialización del LLM (usando OllamaLLM)
    llm = OllamaLLM(model=llm_model_name, temperature=0.0) 

    # 4. La Cadena RAG (Retrieval-Augmented Generation)
    # Se utiliza la sintaxis de diccionario simple para la entrada, que es la más compatible.
    rag_chain = (
        # Paso 1: Mapeo de Entradas (diccionario simple)
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough() # Pasa la pregunta directamente
        }
        # Paso 2: Formatear el Prompt (de un dict a un PromptValue)
        | prompt_template
        # Paso 3: Convertir el PromptValue a string (str) usando RunnableLambda (CRÍTICO para OllamaLLM)
        # Esto soluciona los errores de 'dict' object has no attribute 'replace' y de TypeError.
        | RunnableLambda(lambda prompt_value: prompt_value.text)
        # Paso 4: Pasar la cadena de texto simple al LLM (OllamaLLM)
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# --- FUNCIONES DE SOPORTE DE STREAMLIT ---

@st.cache_resource
def load_tts_component_template():
    """Carga y retorna el contenido del archivo HTML del componente TTS."""
    try:
        with open(TTS_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{TTS_TEMPLATE_FILE}'. Asegúrate de que existe.")
        return None

def generate_tts_button(text_to_speak: str):
    """
    Genera el componente TTS de Streamlit.
    Se eliminó el argumento 'key' para compatibilidad con versiones antiguas de Streamlit.
    """
    tts_html_template = load_tts_component_template()
    if not tts_html_template:
        return

    # Configuración de voz para el componente HTML
    tts_config = {
        "text": text_to_speak,
        "model": TTS_MODEL_NAME,
        "voice": TTS_VOICE_NAME,
        "api_key": API_KEY_GLOBAL 
    }
    
    # Reemplaza el placeholder de la configuración en el HTML
    final_html = tts_html_template.replace('data-tts-config=\'{}\'', f"data-tts-config='{json.dumps(tts_config)}'")
    
    # Renderiza el componente e inicializa JS
    st.components.v1.html(
        f"""
        {final_html}
        <script>
            // This script ensures the component is initialized after rendering
            document.addEventListener('DOMContentLoaded', function() {{
                const container = document.querySelector('.tts-container');
                if (container && window.initTtsComponent) {{
                    window.initTtsComponent(container);
                }}
            }});
            // Fallback for Streamlit re-renders
            if (window.initTtsComponent) {{
                 const container = document.querySelector('.tts-container');
                 if (container) {{
                    window.initTtsComponent(container);
                 }}
            }}
        </script>
        """,
        height=60,
        scrolling=False
        # Se eliminó 'key=key' para compatibilidad
    )

def _get_processed_themes() -> Set[str]:
    """Lee el log para determinar qué temas (carpetas) se han procesado."""
    all_themes = set()
    try:
        with open(PROCESSED_LOG, 'r', encoding='utf-8') as f:
            for line in f: 
                line = line.strip()
                if line and line.startswith(DATA_PATH.replace("./", "")): # ej: libros/Tema/documento.pdf
                    parts = line.split('/')
                    if len(parts) > 1:
                        all_themes.add(parts[1])
    except FileNotFoundError:
        pass
    
    if not all_themes:
        if not os.path.isdir(DATA_PATH):
            all_themes.add("NO_DATA")
    return all_themes

@st.cache_resource
def load_chroma_db(db_path: str, embeddings: HuggingFaceEmbeddings) -> Optional[Chroma]:
    """Carga la base de datos Chroma de forma cacheada y segura."""
    if os.path.isdir(db_path):
        try:
            db = Chroma(persist_directory=db_path, embedding_function=embeddings)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Chroma DB loaded from: {db_path}")
            return db
        except Exception as e:
            st.error(f"Error loading Chroma DB in {db_path}: {e}")
            print(f"Error loading Chroma DB in {db_path}: {e}")
            return None
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Directory not found: {db_path}")
        return None

# --- FUNCIÓN DE LA APLICACIÓN PRINCIPAL ---

def main_app():
    """Función principal que ejecuta la aplicación Streamlit (Solo RAG)."""
    
    st.set_page_config(page_title="RAG Multimodal", layout="wide")
    st.title("📚 Sistema RAG (Generación y Voz TTS)")
    
    # 1. Inicialización de Estado y Embeddings
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'selected_db_path' not in st.session_state:
        st.session_state['selected_db_path'] = CHROMA_DB_PATH_RECURSIVE
    if 'selected_theme' not in st.session_state:
        st.session_state['selected_theme'] = 'ALL_DOCS'
    if 'embeddings' not in st.session_state:
        try:
            st.session_state['embeddings'] = initialize_embeddings()
        except Exception as e:
            st.error(f"No se pudieron cargar los embeddings: {e}. Revisa tu conexión.")
            return

    embeddings = st.session_state['embeddings']
    
    # Mapeo para los selectbox
    db_options = {
        "Recursive (1500/300)": CHROMA_DB_PATH_RECURSIVE,
        "Fixed (1000/200)": CHROMA_DB_PATH_FIXED,
        "Structured (1000/200)": CHROMA_DB_PATH_STRUCTURED,
    }

    # --- Sidebar y Controles ---
    with st.sidebar:
        st.header("Configuración de RAG")

        st.markdown("---")

        # Seleccionar Estrategia DB (Chunking)
        selected_db_name = st.selectbox(
            "Selecciona Estrategia de Indexación (Chunking):",
            list(db_options.keys()),
            index=list(db_options.values()).index(st.session_state['selected_db_path']) 
                  if st.session_state['selected_db_path'] in db_options.values() else 0,
            key='db_selector',
        )
        st.session_state['selected_db_path'] = db_options[selected_db_name]
        
        # Seleccionar Tema
        theme_options_raw = list(_get_processed_themes())
        if "NO_DATA" in theme_options_raw:
             st.error("No se encontraron datos en `./libros/`. Ejecuta `python index_data.py`.")
             return 
             
        theme_options = ["ALL_DOCS"] + [t for t in theme_options_raw if t != "ALL_DOCS"]

        selected_theme = st.selectbox(
            "Filtro por Tema (Carpeta):",
            theme_options,
            index=theme_options.index(st.session_state['selected_theme']) if st.session_state['selected_theme'] in theme_options else 0,
            key='theme_selector',
        )
        st.session_state['selected_theme'] = selected_theme
        st.sidebar.markdown(f"**Tema Seleccionado:** `{selected_theme}`")
    
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Modelo LLM:** `{LLM_MODEL_NAME}` (Vía Ollama)")
        st.sidebar.markdown(f"**Modelo TTS:** `{TTS_MODEL_NAME}` (Voz: `{TTS_VOICE_NAME}`)")


    # --- Carga de Chroma DB y Cadena RAG ---

    current_chain = None
    retriever = None
    is_ready = True 

    db = load_chroma_db(st.session_state['selected_db_path'], embeddings)
    if db is None:
        st.error(f"La base de datos RAG `{selected_db_name}` no se pudo cargar. Asegúrate de que los archivos se hayan indexado correctamente.")
        is_ready = False
    else:
        metadata_filter = {}
        if st.session_state['selected_theme'] != 'ALL_DOCS':
            metadata_filter = {"theme": st.session_state['selected_theme']}
        
        current_chain, retriever = create_rag_chain(
            llm_model_name=LLM_MODEL_NAME, 
            db=db,
            metadata_filter=metadata_filter
        )
    
    st.session_state['current_chain'] = current_chain
    st.session_state['current_retriever'] = retriever

    if not is_ready:
        return 

    # --- INTERFAZ PRINCIPAL DE CHAT Y TTS ---

    # 1. Mostrar Historial de Mensajes y Botones TTS
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "tts_text" in message:
                # El botón TTS se genera aquí para mensajes anteriores
                # Ya no se usa 'key' aquí
                generate_tts_button(message["tts_text"])

    # 2. Manejo de la Entrada del Usuario
    if prompt := st.chat_input("Escribe tu pregunta para los documentos indexados..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Llamar a la cadena (LLM)
        with st.chat_message("assistant"):
            with st.spinner(f"Generando respuesta RAG con estrategia `{selected_db_name}`..."):
                response_container = st.empty()
                
                try:
                    if st.session_state['current_chain'] is None:
                         raise Exception("La cadena RAG no está inicializada.")

                    # Invocar la cadena LLM para obtener la respuesta base
                    # Se pasa el prompt como input único. La cadena se encarga del resto.
                    assistant_response_base = st.session_state['current_chain'].invoke(prompt)
                    
                    # Obtener las fuentes del retriever
                    # Se invoca al retriever por separado para obtener las fuentes que se usaron.
                    retrieved_docs = st.session_state['current_retriever'].invoke(prompt)
                    
                    formatted_sources = "\n".join([
                        f"- {doc.metadata.get('source', 'Fuente desconocida')} (Página {doc.metadata.get('page', 'N/A')}, Tema: {doc.metadata.get('theme', 'N/A')})"
                        for doc in retrieved_docs
                    ])
                    
                    # TTS necesita solo la respuesta del LLM
                    tts_text = assistant_response_base 
                    full_response_text = f"{assistant_response_base}\n\n**Fuentes:**\n{formatted_sources}"

                    # Mostrar la respuesta completa
                    response_container.markdown(full_response_text)
                    
                    # Agregar al historial con el texto para TTS
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response_text,
                        "tts_text": tts_text 
                    })
                    
                    # Generar el botón TTS inmediatamente después de la respuesta
                    # Ya no se usa 'key' aquí
                    generate_tts_button(tts_text)
                    
                except Exception as e:
                    error_msg = f"Error al generar la respuesta RAG. Revisa el log de la terminal. Error: {e}"
                    response_container.error(error_msg)


if __name__ == "__main__":
    main_app()