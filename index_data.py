import os
import shutil
from typing import Set, Tuple
from datetime import datetime
import gc 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma 

# --- CONFIGURACIÓN ---
DATA_PATH = "./libros/"
CHROMA_DB_PATH = "./chroma_db_rag"
PROCESSED_LOG = "./processed_files.txt"
EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2" 
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# La función de inicialización de embeddings ahora es necesaria aquí para la ejecución standalone
def initialize_embeddings():
    """Inicializa y retorna el modelo de embeddings."""
    try:
        # Cargamos el modelo de embeddings.
        return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    except Exception as e:
        print(f"[ERROR CRÍTICO] No se pudo cargar el modelo de embeddings: {e}")
        print("Asegúrate de que el modelo esté disponible localmente o de tener acceso a Internet.")
        raise

def _get_processed_files(log_path: str) -> Set[str]:
    """Lee el log y extrae solo las rutas de los archivos, ignorando las líneas de encabezado/pie de página."""
    processed_files = set()
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Excluir líneas de metadatos 
                if line.startswith('---') or line.startswith('RESUMEN') or line.startswith('DURACIÓN') or line.startswith('FECHA') or line.startswith('ARCHIVOS'):
                    continue
                
                if line.startswith('['):
                    # Formato nuevo iterativo: [DURACIÓN: 0.15s] [TEMA: GENERAL] [PÁGS: 10] ./libros/file.pdf
                    try:
                        path_parts = line.rsplit('] ', 1) 
                        if len(path_parts) > 1:
                            file_path = path_parts[1].strip()
                            processed_files.add(file_path)
                    except:
                        continue 
                else:
                    processed_files.add(line)
    return processed_files


def load_and_index_data(data_path: str, db_path: str, embeddings_model: HuggingFaceEmbeddings):
    """
    Carga, divide y vectoriza documentos de forma ITERATIVA (archivo por archivo).
    Esta función NO devuelve la DB, solo la crea o actualiza en disco, y genera el log.
    
    NOTA DE CORRECCIÓN: Se eliminaron todas las llamadas a `db.persist()` 
    porque este método puede no existir en las versiones recientes de `langchain-chroma`. 
    La persistencia ocurre automáticamente cuando `persist_directory` está configurado.
    """
    print("\n--- FASE 1: INDEXACIÓN INICIADA ---")
    start_time = datetime.now()
    
    # --- PASO 1: CARGAR DB EXISTENTE O INICIAR NUEVA ---
    is_new_db = not os.path.exists(db_path)
    db = None
    all_themes = set()
    
    if not is_new_db:
        try:
            db = Chroma(persist_directory=db_path, embedding_function=embeddings_model)
            print(f"Base de datos encontrada. Cargando desde: {db_path}")
        except Exception as e:
            print(f"Error al cargar la DB existente: {e}. Recreando base de datos.")
            shutil.rmtree(db_path, ignore_errors=True)
            is_new_db = True
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    # --- PASO 2: PROCESAMIENTO ITERATIVO ARCHIVO POR ARCHIVO ---
    processed_files = _get_processed_files(PROCESSED_LOG)
    total_new_chunks = 0
    newly_processed_log_entries = [] 
    
    for root, dirs, files in os.walk(data_path):
        # Determinamos el tema basado en el nombre de la subcarpeta
        theme = os.path.basename(root) if root.rstrip(os.path.sep) != data_path.rstrip(os.path.sep) else "GENERAL"
        all_themes.add(theme)
        
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, '.') 
            clean_path = relative_path.strip()
            
            # Indexar solo si es nuevo y es un PDF
            if clean_path not in processed_files and filename.endswith(".pdf"):
                print(f"\n[INFO] Procesando archivo: {filename} (Tema: {theme})")
                
                file_start_time = datetime.now()
                docs = []
                
                try:
                    # 2.A: CARGAR DOCUMENTOS
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata["theme"] = theme 
                        doc.metadata.setdefault('source', file_path)
                        doc.metadata.setdefault('page', '0')
                            
                    # 2.B: FRAGMENTACIÓN (CHUNKING)
                    file_chunks = text_splitter.split_documents(docs)
                    
                    # 2.C: INDEXACIÓN Y ADICIÓN (SUBIR CHUNKS AL INSTANTE)
                    if not db:
                        # Creamos la DB con el primer batch si no existe
                        db = Chroma.from_documents(documents=file_chunks, embedding=embeddings_model, persist_directory=db_path)
                        print(f"  > [DEBUG] Creación inicial de DB con {len(file_chunks)} chunks.")
                    else:
                        # Añadimos a la DB existente
                        db.add_documents(file_chunks) 
                        print(f"  > [DEBUG] {len(file_chunks)} chunks añadidos a la DB existente.")
                    
                    total_new_chunks += len(file_chunks)
                    
                    # 2.D: REGISTRO Y LIBERACIÓN
                    file_end_time = datetime.now()
                    duration = file_end_time - file_start_time
                    
                    log_entry = (f"[DURACIÓN: {duration.total_seconds():.2f}s] "
                                 f"[TEMA: {theme}] [PÁGS: {len(docs)}] {clean_path}")
                                 
                    newly_processed_log_entries.append(log_entry)
                    print(f"  > Archivo indexado en {duration.total_seconds():.2f}s ({len(file_chunks)} chunks).")
                    
                    # Liberación de memoria explícita (crucial para archivos grandes)
                    del docs 
                    gc.collect() 
                    
                except Exception as e:
                    print(f"  > [ERROR CRÍTICO] Error procesando {filename}. Saltando archivo: {e}")
                    gc.collect() 

    # --- PASO 3: PERSISTENCIA FINAL Y ACTUALIZACIÓN DEL LOG ---
    
    # NOTA: La persistencia (guardado en disco) se maneja automáticamente 
    # por la configuración de `persist_directory` de ChromaDB, por lo que 
    # se eliminan las llamadas explícitas a `db.persist()`.
    if db:
        # Forzamos una liberación de memoria al final para asegurar el guardado
        del db
        gc.collect() 

    # 3.B: Crear y escribir el registro (Solo si hubo nuevos archivos)
    if total_new_chunks > 0:
        end_time = datetime.now()
        duration = end_time - start_time
        
        log_content = [
            "",
            "----------------------------------------------------------------------",
            f"RESUMEN DE CARGA (FUNCIÓN COMPLETA): {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"DURACIÓN TOTAL (Función): {duration.total_seconds():.2f} segundos",
            f"ARCHIVOS PROCESADOS ({len(newly_processed_log_entries)} nuevos):",
            "----------------------------------------------------------------------"
        ]
        
        log_content.extend(newly_processed_log_entries)
        log_content.append("----------------------------------------------------------------------")

        # ESCRITURA FINAL Y CORREGIDA DEL LOG
        try:
            with open(PROCESSED_LOG, 'a', encoding='utf-8') as f:
                f.write('\n'.join(log_content))
            print(f"Registro actualizado en: {PROCESSED_LOG}")
        except Exception as e:
            print(f"[ERROR DE ESCRITURA] No se pudo escribir en {PROCESSED_LOG}: {e}")
    else:
        print("No se detectaron archivos nuevos para indexar. El sistema está actualizado.")


# --- Punto de entrada para ejecución standalone ---
if __name__ == "__main__":
    print("--- INICIANDO SCRIPT DE INDEXACIÓN ---")
    
    try:
        embeddings = initialize_embeddings()
    except Exception:
        print("\nINDEXACIÓN FALLIDA. Revisa el error anterior.")
        exit(1)

    try:
        load_and_index_data(DATA_PATH, CHROMA_DB_PATH, embeddings)
        print("\n--- INDEXACIÓN COMPLETA Y PERSISTIDA. ---")
        print(f"La base de datos RAG está lista en {CHROMA_DB_PATH}.")
    except Exception as e:
        print(f"\n[ERROR FATAL] La indexación falló: {e}")
        exit(1)