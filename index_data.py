import os
import shutil
from typing import Set, Tuple, List
from datetime import datetime
import gc 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_core.documents import Document

# Importar initialize_embeddings y constantes necesarias de rag_utils
from rag_utils import initialize_embeddings, EMBED_MODEL_NAME

# --- CONFIGURACIÓN ---
DATA_PATH = "./libros/"
PROCESSED_LOG = "./processed_files.txt"
# Nueva ruta base clara para el experimento de chunking
CHROMA_BASE_PATH = "./db_chunking" 

# Definición de las tres estrategias de chunking basadas en TIPO DE SPLITTER
CHUNK_STRATEGIES = {
    # La estrategia de uso general se renombró para usar un sufijo claro
    "Recursive (Recomendada)": {
        "db_suffix": "_recursive", 
        "splitter_type": "recursive",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "desc": "RecursiveCharacterTextSplitter. Usa separadores jerárquicos (\\n\\n, \\n, espacio) para mantener la coherencia lógica. Es la más robusta."
    },
    "Fixed Size (Simple)": {
        "db_suffix": "_fixed",
        "splitter_type": "character",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "desc": "CharacterTextSplitter. Divide estrictamente por tamaño de carácter. Ignora la estructura lógica, lo que puede causar fragmentos incoherentes."
    },
    "Structural (Avanzada)": {
        "db_suffix": "_structural",
        "splitter_type": "structural",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "desc": "Simulación Estructural (basada en Recursive). Prioriza separadores largos como títulos y encabezados (\\n\\n\\n, \\n\\n) para dividir por secciones claras, ideal para documentos con estructura fuerte."
    },
}

def get_db_path_for_strategy(strategy_name: str) -> str:
    """
    Genera la ruta única de la DB para una estrategia dada.
    Todas las rutas son únicas y no usan la ruta base original ./chroma_db_rag.
    """
    config = CHUNK_STRATEGIES.get(strategy_name, {})
    suffix = config.get("db_suffix", "_default")
    
    # La ruta de la DB será CHROMA_BASE_PATH + sufijo, e.g., './db_chunking_recursive'
    return f"{CHROMA_BASE_PATH}{suffix}"

def get_splitter_instance(strategy_config: dict):
    """Inicializa y retorna la instancia del splitter de texto según la estrategia."""
    
    size = strategy_config['chunk_size']
    overlap = strategy_config['chunk_overlap']
    splitter_type = strategy_config['splitter_type']
    
    if splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=size, 
            chunk_overlap=overlap, 
            separators=["\n\n", "\n", " ", ""] # Jerarquía por defecto
        )
    
    elif splitter_type == "character":
        # Divide rígidamente por tamaño de carácter.
        # CORRECCIÓN: Se eliminó el parámetro 'chunk_size_size'
        return CharacterTextSplitter(
            chunk_size=size, 
            chunk_overlap=overlap, 
            separator="\n\n", # Usar doble salto de línea como separador por defecto
        )
        
    elif splitter_type == "structural":
        # Simula una división estructural priorizando grandes separadores
        return RecursiveCharacterTextSplitter(
            chunk_size=size, 
            chunk_overlap=overlap, 
            separators=[
                "\n\n\n", # Triple salto de línea 
                "\n\n",   # Doble salto de línea (párrafo nuevo)
                "\n",     # Salto de línea simple
                " ",      # Espacio
                ""        # Caracter
            ]
        )
    
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap) # Fallback

# ----------------------------------------------------------------------
# LÓGICA DE CARGA DE DOCUMENTOS
# ----------------------------------------------------------------------

def _get_files_to_process() -> List[str]:
    """Encuentra todos los archivos PDF en DATA_PATH."""
    all_files = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                all_files.append(os.path.abspath(full_path))
    return all_files

def _load_and_tag_documents(file_paths: List[str]) -> List[Document]:
    """Carga los documentos PDF y añade metadatos de 'source' y 'theme'."""
    all_docs = []
    for file_path in file_paths:
        try:
            relative_path = os.path.relpath(os.path.dirname(file_path), DATA_PATH)
            theme = relative_path if relative_path != '.' else "Sin Tema" 
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["theme"] = theme
                doc.metadata["source"] = file_path
                if 'page' in doc.metadata and isinstance(doc.metadata.get('page'), int):
                    # Se añade +1 para que la página mostrada al usuario sea 1-indexed (más intuitiva)
                    doc.metadata['page'] = doc.metadata['page'] + 1 
                all_docs.append(doc)
            
        except Exception as e:
            print(f"[ERROR DE CARGA] No se pudo cargar o procesar {os.path.basename(file_path)}. Error: {e}")
    return all_docs

# ----------------------------------------------------------------------
# FUNCIÓN DE INDEXACIÓN PRINCIPAL (Ajustada para Múltiples DBs)
# ----------------------------------------------------------------------

def load_and_index_all_strategies(embeddings):
    """
    Ejecuta el pipeline de indexación para cada estrategia de chunking.
    Solo indexa si la DB no existe o está vacía.
    """
        
    all_files_to_load = _get_files_to_process()
    
    if not all_files_to_load:
        print("No se encontraron archivos PDF en el directorio './libros/'. Indexación cancelada.")
        return

    print(f"Cargando y etiquetando documentos ({len(all_files_to_load)} archivos)...")
    all_docs = _load_and_tag_documents(all_files_to_load)
    
    if not all_docs:
        print("No se pudieron cargar documentos. Revisa si los PDFs están corruptos.")
        return

    print(f"Total de páginas cargadas para segmentación: {len(all_docs)}")
    print("----------------------------------------------------------------------")
    
    files_processed_for_log = False
    
    # Itera sobre cada estrategia
    for strategy_name, config in CHUNK_STRATEGIES.items():
        start_time = datetime.now()
        db_path = get_db_path_for_strategy(strategy_name)
        
        # 1. Verificar si la DB ya existe para esta estrategia
        if os.path.exists(db_path) and os.listdir(db_path):
            print(f"[{strategy_name}] DB ya existe en {db_path}. Saltando indexación.")
            files_processed_for_log = True 
            continue
            
        print(f"[{strategy_name}] --- INICIANDO INDEXACIÓN ---")
        print(f"   Tipo de Splitter: {config['splitter_type'].upper()}")
        print(f"   Ruta de DB: {db_path}")
        
        # 2. Aplicar Estrategia de Chunking
        splitter = get_splitter_instance(config)
        chunks = splitter.split_documents(all_docs)
        print(f"   Documentos segmentados en {len(chunks)} fragmentos (chunks).")
        
        # 3. Crear Base de Datos
        try:
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )
            
            duration = datetime.now() - start_time
            print(f"   ✅ Indexación completada. DB guardada en: {db_path}")
            print(f"   Duración: {duration.total_seconds():.2f} segundos.")
            files_processed_for_log = True
        except Exception as e:
            print(f"   [ERROR CRÍTICO] Falló la indexación para {strategy_name}: {e}")
            
    # 4. Actualizar Log de Procesados
    if files_processed_for_log:
        print("\n--- ACTUALIZANDO LOG DE ARCHIVOS PROCESADOS ---")
        log_content = [f"{f} - {datetime.now().isoformat()}" for f in all_files_to_load]
        
        try:
            # Aquí sobrescribimos el log para reflejar que al menos una estrategia se completó con estos archivos
            with open(PROCESSED_LOG, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_content))
            print(f"Registro actualizado en: {PROCESSED_LOG}")
        except Exception as e:
            print(f"[ERROR DE ESCRITURA] No se pudo escribir en {PROCESSED_LOG}: {e}")
    else:
        print("\nTodas las bases de datos ya estaban al día o no hay archivos para procesar. Log no actualizado.")

# --- Punto de entrada para ejecución standalone ---
if __name__ == "__main__":
    print("--- INICIANDO SCRIPT DE INDEXACIÓN MÚLTIPLE (COMPARACIÓN DE SPLITTERS) ---")
    
    try:
        # Nota: La función initialize_embeddings está en rag_utils
        embeddings = initialize_embeddings()
    except Exception:
        print("\nINDEXACIÓN FALLIDA. Revisa el error anterior.")
        exit(1)

    try:
        load_and_index_all_strategies(embeddings)
        print("\n--- PROCESO DE INDEXACIÓN MÚLTIPLE COMPLETO. ---")
        print(f"Bases de datos disponibles para el experimento de chunking en las carpetas que inician con: {CHROMA_BASE_PATH}_*")
    except Exception as e:
        # Esto atrapará errores generales, pero el error de parámetro ya fue corregido en get_splitter_instance
        print(f"\n[ERROR CRÍTICO] La indexación falló en la etapa final: {e}")
        exit(1)