import streamlit as st
import os
import json
import shutil
from typing import Set, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# Importamos la clase OllamaLLM actualizada a través de rag_utils
from rag_utils import initialize_embeddings, create_rag_chain, create_summary_chain
# Importaciones de constantes de ruta y estrategias desde index_data
# Usamos try/except para asegurar que las constantes estén disponibles
try:
    # Intenta importar del archivo index_data.py
    from index_data import CHUNK_STRATEGIES, PROCESSED_LOG, EMBED_MODEL_NAME, get_db_path_for_strategy, DATA_PATH
except ImportError:
    # Si index_data.py no existe o no se puede importar, usamos valores por defecto (ESTO DEBE SER EVITADO)
    CHUNK_STRATEGIES = {"Recursive (Recomendada)": {"desc": "Default strategy"}}
    PROCESSED_LOG = "./processed_files.txt"
    EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
    DATA_PATH = "./libros/"
    def get_db_path_for_strategy(strategy): return f"./chroma_db_rag_{strategy.split(' ')[0].lower()}"


from rag_utils import LLM_MODEL_NAME

# --- CONFIGURACIÓN DE LA APP ---
TTS_MODEL_NAME = "gemini-2.5-flash-preview-tts"
TTS_VOICE_NAME = "Charon" 
TTS_TEMPLATE_FILE = "tts_component.html" 
LLM_MODEL_NAME = "mistral" 

# --- OBTENER API KEY GLOBAL DE CANVAS ---
API_KEY_GLOBAL = os.environ.get('__api_key', '')

# --- FUNCIONES DE LOG Y TEMA ---

def _get_processed_themes() -> Set[str]:
    """
    Lee la estructura de carpetas dentro de DATA_PATH para determinar los temas procesados.
    """
    all_themes = {"TODOS"}
    
    # Buscamos todas las subcarpetas dentro de DATA_PATH
    if os.path.exists(DATA_PATH):
        for item in os.listdir(DATA_PATH):
            full_path = os.path.join(DATA_PATH, item)
            # Si es un directorio y no está oculto (como .DS_Store), lo consideramos un tema
            if os.path.isdir(full_path) and not item.startswith('.'):
                all_themes.add(item)
    
    return all_themes

# --- FUNCIÓN DE CARGA DEL HTML (CACHEADA) ---

@st.cache_resource
def load_tts_html_template(file_path: str) -> str:
    """Carga el contenido del archivo HTML de la plantilla TTS."""
    try:
        if not os.path.exists(file_path):
            # Asegúrate de que el archivo tts_component.html esté presente
            return "Error: Plantilla TTS no encontrada." 
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error al cargar la plantilla HTML del TTS: {e}"


def generate_tts_button(text_to_speak: str, key: str):
    """
    Inyecta un botón y la lógica JS para llamar a la API TTS y reproducir la respuesta.
    Aplica limpieza adicional para caracteres de control JSON problemáticos.
    """
    if not API_KEY_GLOBAL:
        return
    
    # --- CORRECCIÓN CLAVE: LIMPIEZA DE CARACTERES ---
    # 1. Reemplazar saltos de línea con un espacio para evitar problemas de JSON string literal
    # La API TTS no necesita el formato Markdown, solo el texto.
    cleaned_text = text_to_speak.replace('\n', ' ').replace('\r', ' ')
    # 2. Reemplazar múltiples espacios con uno solo
    cleaned_text = ' '.join(cleaned_text.split())
    # 3. Serializar el texto limpio en un JSON string.
    safe_text = json.dumps(cleaned_text)
    
    html_code = load_tts_html_template(TTS_TEMPLATE_FILE) # Usar la constante aquí
    
    if html_code.startswith("Error:"):
        st.error(html_code)
        return
        
    html_code = html_code.replace('__TTS_MODEL_NAME__', TTS_MODEL_NAME)
    html_code = html_code.replace('__TTS_VOICE_NAME__', TTS_VOICE_NAME)
    # INYECCIÓN DE LA API KEY EN EL HTML
    html_code = html_code.replace('const API_KEY_GLOBAL = "";', f'const API_KEY_GLOBAL = "{API_KEY_GLOBAL}";')
    html_code = html_code.replace('__SAFE_TEXT__', safe_text)
    html_code = html_code.replace('__KEY__', key) 
    
    if html_code:
        # Usamos key para que Streamlit sepa que es un componente diferente
        st.components.v1.html(html_code, height=60, scrolling=False)

# --- FUNCIÓN DE CARGA DE LA BASE DE DATOS Y RAG ---

# Importante: el hash_func previene problemas al cargar Chroma
# Se usan los selectores como parte de la clave de caché
@st.cache_resource(hash_funcs={Chroma: lambda _: None, HuggingFaceEmbeddings: lambda _: None})
def load_and_initialize_rag(selected_strategy: str, selected_theme: str):
    """Carga la base de datos vectorial específica y crea las cadenas RAG y Summary."""
    
    llm_status = "Intentando conectar LLM..."
    db_path = get_db_path_for_strategy(selected_strategy)
    
    # 1. Verificar si la base de datos existe
    if not os.path.exists(db_path) or not os.listdir(db_path):
        return None, None, f"❌ DB no encontrada en **{os.path.basename(db_path)}**", llm_status

    try:
        # 2. Inicializar Embeddings (HuggingFace)
        embeddings = initialize_embeddings()
        
        # 3. Cargar la base de datos vectorial
        db = Chroma(
            persist_directory=db_path, 
            embedding_function=embeddings
        )
        db_status = f"✅ Listo ({selected_strategy})"
        
        # 4. Crear la cadena RAG (usando búsqueda simple para estabilidad)
        # El tema se pasa para que el filtro sea aplicado por la cadena
        rag_chain_instance = create_rag_chain(db, LLM_MODEL_NAME, selected_theme)
        
        # 5. Crear la cadena de Resumen 
        summary_chain_instance = create_summary_chain(LLM_MODEL_NAME)
        
        # 6. Verificar si OllamaLLM pudo ser inicializado (si devolvió None)
        if rag_chain_instance is None or summary_chain_instance is None:
            llm_status = "❌ Falló. Verifique el servicio Ollama y el modelo 'mistral'."
            return None, None, db_status, llm_status

        llm_status = "✅ OllamaLLM Inicializado"
        # Devolvemos ambas cadenas
        return rag_chain_instance, summary_chain_instance, db_status, llm_status
    
    except Exception as e:
        db_status = f"❌ Error de Chroma/Embeddings: {e}"
        # Devolvemos None en caso de fallo de DB
        return None, None, db_status, llm_status

# --- MANEJO DE LA CONSULTA ---

def handle_user_input(prompt, mode):
    """Maneja la lógica de la consulta y actualiza el historial de chat."""
    
    # Se agrega el mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Verificación de que las cadenas están disponibles
    if st.session_state.rag_chain is None or st.session_state.summary_chain is None:
        st.error("El sistema RAG no está inicializado. Verifica los errores en la barra lateral.")
        st.session_state.messages.append({"role": "assistant", "content": "Error: El sistema RAG no está listo."})
        return
        
    with st.chat_message("assistant"):
        answer = ""
        sources = "N/A"
        
        if mode == "RAG (Búsqueda Documental)":
            
            with st.spinner(f"Consultando DB ({st.session_state.selected_strategy}) y generando respuesta..."):
                try:
                    rag_result = st.session_state.rag_chain.invoke(prompt)
                    answer = rag_result["answer"]
                    sources = rag_result["sources"].strip() 
                        
                except Exception as e:
                    answer = f"Error al ejecutar la cadena RAG: {e}. Revisa la consola para más detalles."
                    sources = "N/A"
                
                # Renderizar la respuesta del LLM
                st.markdown(answer)
                
                # Renderizar las fuentes
                if sources and sources != "N/A" and sources.strip():
                    st.info(f"**Fuentes encontradas (Estrategia: {st.session_state.selected_strategy} | Tema: {st.session_state.selected_theme}):**\n{sources}")
            
        elif mode == "Resumen (Summary)":
            with st.spinner(f"Generando resumen (Summary/{LLM_MODEL_NAME})..."):
                try:
                    answer = st.session_state.summary_chain.invoke(prompt)
                except Exception as e:
                    answer = f"Error al ejecutar la cadena de resumen: {e}. Revisa la consola para más detalles."
                
                sources = "N/A (Modo Resumen)"
                
                st.markdown("**Resumen generado:**")
                st.markdown(answer)
        
        # Botón TTS (se muestra después de la respuesta)
        generate_tts_button(answer, f"tts_{len(st.session_state.messages)}")

    # Guardar el mensaje del asistente al final
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources,
        "tts_text": answer,
        "mode": mode, # Guardamos el modo para la correcta visualización
        "strategy": st.session_state.selected_strategy, # Guardamos la estrategia usada
        "theme": st.session_state.selected_theme
    })


# --- INTERFAZ DE USUARIO ---

def main_app():
    """Define la estructura de la aplicación Streamlit."""
    st.set_page_config(
        page_title="RAG Experiment: Chunking Types & Ollama/Mistral", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    # Inicialización de estados de sesión CRUCIALES
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    available_themes = sorted(list(_get_processed_themes()))
    if 'selected_theme' not in st.session_state:
        st.session_state.selected_theme = available_themes[0] if available_themes else 'TODOS'
    
    strategy_keys = list(CHUNK_STRATEGIES.keys())
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = strategy_keys[0] if strategy_keys else 'Recursive (Recomendada)'

    if 'selected_mode' not in st.session_state:
        st.session_state.selected_mode = "RAG (Búsqueda Documental)"

    # --- BARRA LATERAL (DEBUG Y CONFIGURACIÓN) ---
    
    # Estados temporales para detectar cambios y forzar recarga
    current_strategy = st.session_state['selected_strategy']
    current_theme = st.session_state['selected_theme']
    
    with st.sidebar:
        st.title("Configuración y Estado RAG")
        
        # 1. ESTRATEGIA CHUNKING SELECTOR
        strategy_options = list(CHUNK_STRATEGIES.keys())
        default_index_strategy = strategy_options.index(current_strategy) if current_strategy in strategy_options else 0
        
        selected_strategy = st.selectbox(
            "Selecciona la Estrategia de Segmentación:",
            options=strategy_options,
            index=default_index_strategy,
            key='strategy_selector_input',
            help="Compara cómo la división de texto (Fija, Recursiva, Estructural) afecta el resultado."
        )
        # Si la estrategia cambia, actualizamos el estado, limpiamos y recargamos
        if current_strategy != selected_strategy:
            st.session_state['selected_strategy'] = selected_strategy
            st.session_state.messages = [] # Limpiar historial al cambiar de DB
            st.cache_resource.clear() 
            st.rerun() 

        db_path_info = os.path.basename(get_db_path_for_strategy(selected_strategy))
        st.markdown(f"**Ruta de DB:** `{db_path_info}/`")
        st.markdown(f"**Tipo:** `{CHUNK_STRATEGIES.get(selected_strategy, {'desc': 'N/A'})['desc']}`")
        
        st.markdown("---")
        
        # 2. TEMA SELECTOR
        available_themes = sorted(list(_get_processed_themes()))
        theme_options = available_themes

        try:
             default_theme_index = theme_options.index(current_theme)
        except ValueError:
             default_theme_index = 0

        selected_theme = st.selectbox(
            "Filtrar documentos por tema:",
            options=theme_options,
            index=default_theme_index,
            key='theme_selector_input',
            help=f"Los temas disponibles se detectan en la estructura de carpetas de `{DATA_PATH}`."
        )
        # Si el tema cambia, actualizamos el estado, limpiamos y recargamos
        if current_theme != selected_theme:
            st.session_state['selected_theme'] = selected_theme
            st.session_state.messages = [] # Limpiar historial al cambiar de filtro
            st.cache_resource.clear() 
            st.rerun() 
        
        st.markdown("---")
        
        # 3. MODO SELECTOR
        selected_mode = st.selectbox(
            "Seleccionar modo de operación:",
            options=["RAG (Búsqueda Documental)", "Resumen (Summary)"],
            key="mode_selector_input",
            index=0 if st.session_state.selected_mode == "RAG (Búsqueda Documental)" else 1,
            help="El modo RAG busca en tus documentos. El modo Resumen usa el texto de la caja de entrada para resumir."
        )
        st.session_state.selected_mode = selected_mode

        # Initialization of RAG and DB (Depende de la estrategia y el tema seleccionados)
        # Lo que asegura que la caché se invalida si cambian (aunque reran() ya lo hace)
        rag_chain, summary_chain, db_status, llm_status = load_and_initialize_rag(selected_strategy, selected_theme)
        
        # Actualizar estados de la sesión
        st.session_state.rag_chain = rag_chain
        st.session_state.summary_chain = summary_chain
        st.session_state.db_status = db_status
        st.session_state.llm_status = llm_status
        
        st.markdown("---")
        st.markdown("### 🛠️ Diagnóstico del Sistema")
        
        key_status = '✅ Cargada' if API_KEY_GLOBAL else '❌ No Cargada/Vacía'
        st.markdown(f"**Clave API (TTS) Estado:** `{key_status}`")
        
        st.markdown(f"**Estado de la Base de Datos (Chroma):** `{st.session_state.db_status}`")
        st.markdown(f"**Estado de conexión LLM (OllamaLLM):** `{st.session_state.llm_status}`")
        st.markdown("---")
        st.markdown(f"**Modelo LLM:** `{LLM_MODEL_NAME}`")
        st.markdown(f"**Modelo Embeddings:** `{EMBED_MODEL_NAME}`")
        st.markdown(f"**Modelo TTS:** `{TTS_MODEL_NAME}` (Voz: `{TTS_VOICE_NAME}`)")
        
        st.subheader("Instrucciones Clave")
        st.markdown("""
        1. **Indexación:** Ejecuta `python index_data.py` para crear las DBs faltantes.
        2. **Ollama:** Asegúrate de que el modelo `mistral` esté disponible y el servicio esté corriendo.
        """)
        
        if st.button("Limpiar Historial de Chat"):
            st.session_state.messages = []
            st.cache_resource.clear()
            st.rerun()

    # --- INTERFAZ PRINCIPAL DE CHAT ---

    st.title("Laboratorio RAG: Comparación de Tipos de Segmentación")
    st.markdown(f"**DB Activa:** `{selected_strategy}` ({db_path_info}) | **Filtro de Tema:** `{selected_theme}` | **Modo:** `{selected_mode}`")

    # Determinar si la app está lista
    is_indexed = rag_chain is not None and summary_chain is not None

    if not is_indexed:
        # Mostrar el error que viene de la función de carga
        st.error(f"El sistema RAG no está inicializado. Por favor, verifica la barra lateral. Motivo: {db_status} o {llm_status}")
        return

    # 1. Mostrar Historial de Mensajes y Botones TTS
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "tts_text" in message:
                generate_tts_button(message["tts_text"], f"history_tts_{idx}")
            
            # Mostrar fuentes solo si:
            # a) El mensaje fue generado en modo RAG.
            # b) Las fuentes existen y no son "N/A" o vacías.
            if (message["role"] == "assistant" and 
                message.get("mode") == "RAG (Búsqueda Documental)" and 
                message.get("sources", "N/A").strip() not in ["N/A", ""]):
                 
                 # Usamos la estrategia y tema guardados en el mensaje, no los actuales de la sesión
                 msg_strategy = message.get("strategy", "N/A")
                 msg_theme = message.get("theme", "N/A")
                 
                 st.info(f"**Fuentes encontradas (Estrategia usada: {msg_strategy} | Tema: {msg_theme}):**\n{message['sources']}")

    # 2. Text input for new query
    if is_indexed:
        prompt = st.chat_input("Escribe tu pregunta o el texto a resumir aquí...")
        if prompt:
            handle_user_input(prompt, st.session_state.selected_mode)

if __name__ == "__main__":
    main_app()