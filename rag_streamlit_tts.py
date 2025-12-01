import streamlit as st
import os
import json
import shutil
from typing import Set, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader 

# Importaciones de las funciones centrales
# initialize_embeddings y create_rag_chain deben estar en rag_utils
from rag_utils import initialize_embeddings, create_rag_chain
# Importaciones de constantes de ruta desde index_data
from index_data import DATA_PATH, CHROMA_DB_PATH, EMBED_MODEL_NAME

# --- CONFIGURACIÓN DE LA APP ---
TTS_MODEL_NAME = "gemini-2.5-flash-preview-tts"
TTS_VOICE_NAME = "Charon" 
TTS_TEMPLATE_FILE = "tts_component.html" 
# CORRECCIÓN DE LLM: Usando el nombre de modelo exacto solicitado por el usuario.
LLM_MODEL_NAME = "mistral" 

# --- CONSTANTES GLOBALES (Necesarias para la lógica de log) ---
PROCESSED_LOG = "./processed_files.txt"

# --- OBTENER API KEY GLOBAL DE CANVAS ---
API_KEY_GLOBAL = os.environ.get('__api_key', '')

def _get_processed_themes() -> Set[str]:
    """
    Lee el log para determinar qué temas (carpetas) se han procesado.
    Esto es necesario para el filtro incluso si la indexación es externa.
    """
    all_themes = {"GENERAL"}
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('[DURACIÓN:'):
                    # Extraer el tema del registro
                    try:
                        start_idx = line.find('[TEMA: ') + len('[TEMA: ')
                        end_idx = line.find(']', start_idx)
                        if start_idx != -1 and end_idx != -1:
                            theme = line[start_idx:end_idx].strip()
                            all_themes.add(theme)
                    except:
                        continue
    
    # Recorrer la carpeta de datos para encontrar temas, si no hay log aún
    if not os.path.exists(PROCESSED_LOG):
         for root, dirs, files in os.walk(DATA_PATH):
            theme = os.path.basename(root) if root.rstrip(os.path.sep) != DATA_PATH.rstrip(os.path.sep) else "GENERAL"
            all_themes.add(theme)

    return all_themes


# ----------------------------------------------------------------------
# FUNCIÓN DE CARGA DEL HTML (CACHEADA)
# ----------------------------------------------------------------------

@st.cache_resource
def load_tts_html_template(file_path: str) -> str:
    """Carga el contenido del archivo HTML de la plantilla TTS."""
    try:
        if not os.path.exists(file_path):
            return f"Error: Plantilla TTS '{file_path}' no encontrada."
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error al cargar la plantilla HTML del TTS: {e}")
        return ""


def generate_tts_button(text_to_speak: str, key: str):
    """Inyecta un botón y la lógica JS para llamar a la API TTS y reproducir la respuesta."""
    safe_text = json.dumps(text_to_speak)
    html_code = load_tts_html_template(TTS_TEMPLATE_FILE)
    
    if html_code.startswith("Error:"):
        st.error(html_code)
        return
        
    html_code = html_code.replace('__TTS_MODEL_NAME__', TTS_MODEL_NAME)
    html_code = html_code.replace('__TTS_VOICE_NAME__', TTS_VOICE_NAME)
    html_code = html_code.replace('const API_KEY_GLOBAL = "";', f'const API_KEY_GLOBAL = "{API_KEY_GLOBAL}";')
    html_code = html_code.replace('__SAFE_TEXT__', safe_text)
    html_code = html_code.replace('__KEY__', key) 
    
    if html_code:
        st.components.v1.html(html_code, height=60, scrolling=False)


# ----------------------------------------------------------------------
# FUNCIÓN DE CARGA AUTOMÁTICA Y CACHEADA (al iniciar)
# ----------------------------------------------------------------------

@st.cache_resource
def load_cached_data() -> Tuple[Chroma | None, Set[str]]:
    """
    SÓLO CARGA la base de datos de vectores existente. 
    No intenta indexar.
    """
    embeddings_model = initialize_embeddings()
    all_themes = _get_processed_themes()
    
    if not os.path.exists(CHROMA_DB_PATH):
        # La DB no existe
        return None, all_themes
    
    try:
        # La DB existe, la cargamos.
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings_model)
        print(f"Base de datos RAG cargada con éxito desde: {CHROMA_DB_PATH}")
        return db, all_themes
        
    except Exception as e:
        print(f"[ERROR DE CARGA] No se pudo cargar la DB: {e}")
        # Si la carga falla (DB corrupta), devolvemos None
        return None, all_themes


def handle_query(question: str, db: Chroma, selected_theme: str):
    """Ejecuta la consulta RAG y maneja la lógica de chat, visualización y TTS."""
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner(f"Consultando RAG en el tema **{selected_theme}**..."):
        try:
            # Usamos el LLM_MODEL_NAME = "mistral"
            rag_chain = create_rag_chain(db, LLM_MODEL_NAME, selected_theme)
            response = rag_chain.invoke(question)
            full_answer = response['answer']
            
            sources_raw = response.get('sources', '').strip()
            sources_list = [s.strip() for s in sources_raw.split('\n') if s.strip()]
            
            with st.chat_message("assistant"):
                st.markdown(full_answer)
                generate_tts_button(full_answer, f"tts_{len(st.session_state.messages)}")
                
                st.markdown(f"**Fuentes (Filtro: {selected_theme})**")
                
                if sources_list:
                    st.markdown('\n'.join([f"- {s}" for s in sources_list]))
                else:
                    st.markdown("_(No se pudo recuperar información de origen específica para esta respuesta.)_")


            formatted_response_for_history = f"{full_answer}\n\n---\n\n### Fuentes Utilizadas\n"
            if sources_list:
                formatted_response_for_history += '\n'.join([f"- {s}" for s in sources_list])
            else:
                formatted_response_for_history += "No se recuperaron fuentes."
                
            st.session_state.messages.append({"role": "assistant", "content": formatted_response_for_history, "tts_text": full_answer})
            
        except Exception as e:
            error_msg = f"Ocurrió un error durante la consulta: {e}. Revisa la configuración de tu API Key y asegúrate de que el script `rag_utils.py` utilice el modelo Mistral correctamente."
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": f"ERROR: {e}"})
            print(error_msg)


def main():
    st.set_page_config(page_title="Sistema RAG Documental Chat con Voz", layout="wide")

    st.title("💬 Chat RAG Documental (TTS Habilitado)")
    st.markdown("""
        **Modo Chat:** Haz preguntas. La información se recupera de tus PDFs. **Nota:** La indexación de nuevos documentos debe hacerse ejecutando `python index_data.py` en la terminal.
    """)

    # --- INICIALIZACIÓN DE ESTADO ---
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'selected_theme' not in st.session_state:
        st.session_state['selected_theme'] = 'TODOS'

    # --- INICIO AUTOMÁTICO DE CARGA DE DATOS (Solo Lectura) ---
    is_indexed = False
    db = None
    available_themes = set()
    try:
        db, available_themes = load_cached_data()
        st.session_state['db'] = db
        st.session_state['available_themes'] = sorted(list(available_themes))
        is_indexed = (db is not None)
    except Exception as e:
        st.error("Error al cargar la base de datos en caché. Por favor, verifica tu configuración y dependencias.")
        is_indexed = False

    # --- SIDEBAR (Barra Lateral) ---
    st.sidebar.header("Control y Filtros")
    
    # DIAGNÓSTICO DE CLAVE API 
    key_status = '✅ Cargada' if API_KEY_GLOBAL else '❌ No Cargada/Vacía'
    st.sidebar.markdown(f"**Clave API (TTS) Estado:** `{key_status}`")
    if API_KEY_GLOBAL:
        st.sidebar.code(f"Clave (últimos 4 dígitos): {API_KEY_GLOBAL[-4:]}")
    else:
        st.sidebar.warning("El TTS no funcionará sin la Clave API.")
    
    # Mensaje fijo sobre la indexación
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Estado de la Base de Datos")
    st.sidebar.markdown(f"**Estado DB:** {'✅ Cargada y Lista' if is_indexed else '❌ DB no encontrada'}")
    if not is_indexed:
        st.sidebar.info("Para indexar nuevos documentos, ejecuta en tu terminal:\n`python index_data.py`")

    
    if is_indexed and db is not None:
        available_themes_list = st.session_state.get('available_themes', [])
        theme_options = ['TODOS'] + available_themes_list
        
        selected_theme = st.sidebar.selectbox(
            "Filtrar Documentos por Tema:",
            options=theme_options,
            index=theme_options.index(st.session_state['selected_theme']) if st.session_state['selected_theme'] in theme_options else 0,
            key='theme_selector',
            help="Selecciona un tema para restringir la búsqueda en la DB."
        )
        st.session_state['selected_theme'] = selected_theme
        st.sidebar.markdown(f"**Tema Seleccionado:** `{selected_theme}`")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Modelo LLM:** `{LLM_MODEL_NAME}`")
    st.sidebar.markdown(f"**Modelo TTS:** `{TTS_MODEL_NAME}` (Voz: `{TTS_VOICE_NAME}`)")
    st.sidebar.markdown(f"**Documentos monitoreados en:** `./libros/`")

    # --- INTERFAZ PRINCIPAL DE CHAT ---

    if not is_indexed:
        st.error("La base de datos RAG no se pudo cargar. Por favor, asegúrate de que exista y esté actualizada ejecutando `python index_data.py`.")
        return

    # 1. Mostrar Historial de Mensajes y Botones TTS
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "tts_text" in message:
                generate_tts_button(message["tts_text"], f"history_tts_{idx}")

    # 2. Capturar Nueva Pregunta del Usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        handle_query(prompt, st.session_state['db'], st.session_state['selected_theme'])


if __name__ == "__main__":
    main()