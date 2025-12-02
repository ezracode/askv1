import os
from typing import Set, Tuple, List
from datetime import datetime
import gc 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma 
from langchain_core.documents import Document

# --- CONFIGURACIÓN DE RUTAS Y MODELOS ---
DATA_PATH = "./libros/"
PROCESSED_LOG = "./processed_files.txt"
EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2" 

# Rutas de bases de datos Chroma (¡CRÍTICO: Deben ser consistentes!)
CHROMA_DB_PATH_RECURSIVE = "./chroma_db_rag_recursive"
CHROMA_DB_PATH_FIXED = "./chroma_db_rag_fija"
CHROMA_DB_PATH_STRUCTURED = "./chroma_db_rag_estructurada" 

# Configuración de chunking
CHUNK_SIZE_REC = 1500
CHUNK_OVERLAP_REC = 300
CHUNK_SIZE_FIXED_STRUCT = 1000 
CHUNK_OVERLAP_FIXED_STRUCT = 200

def initialize_embeddings():
    """Inicializa y retorna el modelo de embeddings."""
    try:
        return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load embeddings model: {e}")
        raise

def _get_processed_files(log_path: str) -> Set[str]:
    """Lee el log y extrae solo las rutas de los archivos."""
    processed_files = set()
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f: 
                line = line.strip()
                if line and not line.startswith("---") and not line.startswith("DURACIÓN") and line.endswith('.pdf'):
                    processed_files.add(line.replace('\\', '/').replace('./', '')) 
    except FileNotFoundError:
        pass
    return processed_files

def _load_new_documents(data_path: str, processed_files: Set[str]) -> Tuple[List[Document], List[str]]:
    """Carga documentos nuevos que no están en el log."""
    new_documents = []
    newly_processed_paths = []
    
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                log_path = os.path.relpath(file_path, start='./').replace('\\', '/')
                
                relative_dir = os.path.relpath(root, data_path)
                theme_name = relative_dir if relative_dir != '.' else "UNTITLED"
                
                if log_path not in processed_files:
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        
                        for doc in docs:
                            doc.metadata['theme'] = theme_name
                            
                        new_documents.extend(docs)
                        newly_processed_paths.append(log_path) 
                        print(f"   [Cargando] {file} (Tema: {theme_name})")
                        
                    except Exception as e:
                        print(f"   [ERROR] No se pudo cargar {file}: {e}")
                else:
                    print(f"   [Omitiendo] {file} (Ya procesado)")
    
    return new_documents, newly_processed_paths

def load_and_index_data(chroma_db_path: str, embeddings: HuggingFaceEmbeddings, new_documents: List[Document], strategy: str):
    """Divide, vectoriza e indexa los documentos en la base de datos Chroma especificada."""
    if not new_documents:
        print(f"   -> No hay documentos nuevos para indexar en {chroma_db_path}.")
        return 0
        
    print(f"   -> Indexando {len(new_documents)} documentos en {chroma_db_path} con estrategia '{strategy}'...")

    if strategy == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_REC, chunk_overlap=CHUNK_OVERLAP_REC)
    elif strategy == "fixed": 
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE_FIXED_STRUCT, chunk_overlap=CHUNK_OVERLAP_FIXED_STRUCT)
    elif strategy == "structured":
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE_FIXED_STRUCT, chunk_overlap=CHUNK_OVERLAP_FIXED_STRUCT, separator="\n\n")
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_REC, chunk_overlap=CHUNK_OVERLAP_REC)

    chunks = text_splitter.split_documents(new_documents)
    
    db = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    db.add_documents(chunks) 
    
    print(f"   [Éxito] {len(chunks)} chunks vectorizados y guardados en {chroma_db_path}.")
    
    del db
    gc.collect() 
    return len(chunks)

def run_indexing():
    """Función principal para coordinar la carga y las tres indexaciones."""
    start_time = datetime.now()
    print(f"--- FASE 1: CARGANDO DOCUMENTOS ---")
    
    processed_files = _get_processed_files(PROCESSED_LOG)
    new_documents, newly_processed_paths = _load_new_documents(DATA_PATH, processed_files)

    if newly_processed_paths:
        print(f"\nSe encontraron {len(newly_processed_paths)} archivos nuevos (Total de páginas: {len(new_documents)}).")
        
        try:
            embeddings = initialize_embeddings()
        except Exception as e:
            print(f"\n[ERROR] No se pudo inicializar los Embeddings. Abortando indexación: {e}")
            return 

        # Indexación en las tres bases de datos
        load_and_index_data(CHROMA_DB_PATH_RECURSIVE, embeddings, new_documents, "recursive")
        load_and_index_data(CHROMA_DB_PATH_FIXED, embeddings, new_documents, "fixed")
        load_and_index_data(CHROMA_DB_PATH_STRUCTURED, embeddings, new_documents, "structured")
        
        # Actualizar el LOG
        duration = datetime.now() - start_time
        log_content = [
            f"\n\n--- REGISTRO DE INDEXACIÓN ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---",
            f"DURACIÓN TOTAL (Función): {duration.total_seconds():.2f} segundos",
            f"ARCHIVOS PROCESADOS ({len(newly_processed_paths)} nuevos):"
        ]
        log_content.extend(newly_processed_paths)
        log_content.append("----------------------------------------------------------------------")

        try:
            with open(PROCESSED_LOG, 'a', encoding='utf-8') as f:
                f.write('\n'.join(log_content))
            print(f"\nRegistro actualizado en: {PROCESSED_LOG}")
        except Exception as e:
            print(f"[ERROR DE ESCRITURA] No se pudo escribir en {PROCESSED_LOG}: {e}")
    else:
        print("No se detectaron archivos nuevos para indexar. El sistema está actualizado.")
        print(f"DURACIÓN TOTAL (Chequeo): {(datetime.now() - start_time).total_seconds():.2f} segundos")


if __name__ == "__main__":
    print("--- INICIANDO SCRIPT DE INDEXACIÓN ---\n")
    try:
        run_indexing() 
        print("\n--- INDEXACIÓN COMPLETA Y PERSISTIDA. ---")
    except Exception as e:
        print(f"\nINDEXACIÓN FALLIDA. Revisa el error: {e}")
        exit(1)