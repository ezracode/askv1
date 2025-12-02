import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

# LangChain Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

# Importaciones de constantes de ruta desde el script de indexación
try:
    from index_data import (
        EMBED_MODEL_NAME, 
        CHROMA_DB_PATH_RECURSIVE, 
        CHROMA_DB_PATH_FIXED, 
        CHROMA_DB_PATH_STRUCTURED,
        initialize_embeddings
    )
except ImportError:
    print("Error: No se pudo importar 'index_data.py'. Asegúrate de que el archivo existe y las constantes están definidas.")
    exit()

# --- CONFIGURACIÓN CENTRAL ---
LLM_MODEL_NAME = "mistral" 
EVAL_DATA_FILE = "eval_data.json"
API_KEY_GLOBAL = os.environ.get('__api_key', '') 

# --- UTILIDADES DE RAG YA PROBADAS ---

def format_docs(docs: List[Document]) -> str:
    """Format the documents content into a single string for the prompt."""
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def load_chroma_db(db_path: str, embeddings: HuggingFaceEmbeddings) -> Optional[Chroma]:
    """Loads the Chroma database securely."""
    if os.path.isdir(db_path):
        try:
            return Chroma(persist_directory=db_path, embedding_function=embeddings)
        except Exception as e:
            print(f"Error al cargar Chroma DB en {db_path}: {e}")
            return None
    else:
        print(f"Directorio no encontrado: {db_path}")
        return None

def create_rag_chain(llm_model_name: str, db: Chroma, metadata_filter: dict):
    """Creates and returns the stable RAG chain using the specified LLM and retriever."""
    
    # 1. Construcción dinámica de search_kwargs para evitar filtros vacíos
    search_kwargs = {"k": 6}
    if metadata_filter:
        # Añade el filtro SOLO si metadata_filter no está vacío.
        search_kwargs["filter"] = metadata_filter

    retriever = db.as_retriever(
        search_kwargs=search_kwargs
    )
    
    template_str = """
    Eres un asistente de recuperación de información experto.
    Usa SÓLO el siguiente contexto para responder a la pregunta.
    Si la respuesta no se encuentra en el contexto proporcionado, responde honestamente que no tienes la información disponible.

    Contexto: {context}
    Pregunta: {question}

    Respuesta con fuentes:
    """
    prompt_template = PromptTemplate.from_template(template_str)
    
    llm = OllamaLLM(model=llm_model_name, temperature=0.0) 

    # RAG Chain con la sintaxis estable de LCEL (diccionario + RunnableLambda)
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough() 
        }
        | prompt_template
        | RunnableLambda(lambda prompt_value: prompt_value.text) # CRÍTICO para compatibilidad
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# --- LÓGICA DE EVALUACIÓN (LLM-AS-A-JUDGE) ---

def create_evaluation_chain(prompt_template: str):
    """Creates a basic LLM evaluation chain."""
    llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=0.0) 
    
    eval_prompt = PromptTemplate.from_template(prompt_template)
    
    # Función de limpieza para asegurar que la respuesta del LLM es un número
    def clean_and_convert_to_float(text):
        cleaned_text = text.strip().replace('\n', '').replace('\r', '')
        try:
            return float(cleaned_text)
        except ValueError:
            print(f"  [ADVERTENCIA] El LLM evaluador no devolvió un float limpio: '{text.strip()}'")
            return 0.0

    eval_chain = (
        eval_prompt
        | RunnableLambda(lambda prompt_value: prompt_value.text)
        | llm
        | StrOutputParser()
        | RunnableLambda(clean_and_convert_to_float) 
    )
    return eval_chain

# Templates para las métricas (LLM-as-a-Judge)

FAITHFULNESS_TEMPLATE = """
Evalúa la fidelidad de la respuesta generada con respecto al contexto proporcionado.
La fidelidad mide si la respuesta se basa ÚNICAMENTE en la evidencia proporcionada en el contexto.

Contexto: {context}
Respuesta Generada: {answer}

Reglas:
1. Si TODAS las afirmaciones en la Respuesta Generada se pueden verificar directamente en el Contexto, devuelve 1.0.
2. Si algunas afirmaciones en la Respuesta Generada NO se encuentran o no están soportadas por el Contexto, devuelve 0.0.
3. Si la Respuesta Generada es una negación o no puede ser evaluada, devuelve 0.0.

Devuelve SÓLO el número (0.0 o 1.0).
"""

CONTEXT_RELEVANCE_TEMPLATE = """
Evalúa la relevancia de los fragmentos de contexto proporcionados para responder la pregunta.

Pregunta: {question}
Contexto: {context}

Reglas:
1. Si la mayor parte del Contexto es directamente útil y relevante para responder la Pregunta, devuelve 1.0.
2. Si el Contexto contiene información significativamente irrelevante o es demasiado breve para ser útil, devuelve 0.0.

Devuelve SÓLO el número (0.0 o 1.0).
"""

ANSWER_RELEVANCE_TEMPLATE = """
Evalúa la relevancia de la respuesta generada con respecto a la pregunta original.

Pregunta: {question}
Respuesta Generada: {answer}

Reglas:
1. Si la Respuesta Generada aborda directamente la Pregunta y es informativa, devuelve 1.0.
2. Si la Respuesta Generada es tangencial, incompleta, o evade la pregunta, devuelve 0.0.

Devuelve SÓLO el número (0.0 o 1.0).
"""

def evaluate_metrics(
    question: str, 
    answer: str, 
    context: str
) -> Dict[str, float]:
    """Calculates the three RAG metrics using the LLM as a judge."""
    
    results = {}
    
    # 1. Faithfulness (Fidelidad)
    faithfulness_chain = create_evaluation_chain(FAITHFULNESS_TEMPLATE)
    results['faithfulness'] = faithfulness_chain.invoke({"context": context, "answer": answer})
    
    # 2. Context Relevance (Relevancia del Contexto)
    context_relevance_chain = create_evaluation_chain(CONTEXT_RELEVANCE_TEMPLATE)
    results['context_relevance'] = context_relevance_chain.invoke({"question": question, "context": context})

    # 3. Answer Relevance (Relevancia de la Respuesta)
    answer_relevance_chain = create_evaluation_chain(ANSWER_RELEVANCE_TEMPLATE)
    results['answer_relevance'] = answer_relevance_chain.invoke({"question": question, "answer": answer})
    
    return results

# --- FUNCIÓN PRINCIPAL DE EVALUACIÓN ---

def main_evaluation():
    
    print("--- 🔬 Iniciando Evaluación de Métricas RAG (LLM: Mistral/Ollama) ---")
    
    # 1. Cargar datos de evaluación
    if not os.path.exists(EVAL_DATA_FILE):
        print(f"\nERROR: Archivo de datos de evaluación no encontrado: '{EVAL_DATA_FILE}'")
        print("Por favor, crea el archivo y añade preguntas relacionadas con tus documentos.")
        return

    try:
        with open(EVAL_DATA_FILE, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\nERROR FATAL AL CARGAR EL JSON: {e}")
        print("El archivo 'eval_data.json' contiene un error de formato. Reemplázalo con la versión limpia si el problema persiste.")
        return
    
    if not eval_data:
        print("\nERROR: El archivo de datos de evaluación está vacío.")
        return

    # 2. Configurar bases de datos y LLM
    embeddings = initialize_embeddings()
    db_paths = {
        "Recursive": CHROMA_DB_PATH_RECURSIVE,
        "Fixed": CHROMA_DB_PATH_FIXED,
        "Structured": CHROMA_DB_PATH_STRUCTURED,
    }
    
    all_results = {}

    for db_name, db_path in db_paths.items():
        print(f"\n=======================================================")
        print(f"|  Evaluando estrategia: {db_name}  |")
        print(f"=======================================================")
        
        db = load_chroma_db(db_path, embeddings)
        if db is None:
            print(f"Saltando evaluación para {db_name} (DB no cargada).")
            continue

        # metadata_filter está vacío aquí, lo cual es manejado por create_rag_chain
        metadata_filter = {} 
        
        try:
            # Creamos la cadena RAG por DB
            rag_chain, retriever = create_rag_chain(LLM_MODEL_NAME, db, metadata_filter)
        except Exception as e:
            print(f"Error al crear la cadena RAG para {db_name}: {e}")
            continue

        db_results = []
        
        # 3. Iterar sobre los datos de prueba
        for i, item in enumerate(eval_data):
            question = item['question']
            
            print(f"\n--- Pregunta {i+1}: {question[:50]}...")
            
            # Generación de la Respuesta RAG
            try:
                answer = rag_chain.invoke(question)
            except Exception as e:
                print(f"  [ERROR] Falló la generación RAG para la pregunta {i+1}: {e}")
                answer = "Error de generación."

            # Recuperación del Contexto (para evaluación)
            retrieved_docs = retriever.invoke(question)
            context = format_docs(retrieved_docs)
            
            # Evaluación de Métricas
            try:
                metrics = evaluate_metrics(question, answer, context)
            except Exception as e:
                print(f"  [ERROR] Falló la evaluación de métricas (LLM Judge) para la pregunta {i+1}: {e}")
                metrics = {'faithfulness': 0.0, 'context_relevance': 0.0, 'answer_relevance': 0.0}
            
            # Almacenar resultados
            db_results.append({
                "question": question,
                "answer": answer,
                "context": context,
                "metrics": metrics
            })
            
            print(f"  Fidelidad: {metrics['faithfulness']:.2f}, Relevancia Contexto: {metrics['context_relevance']:.2f}, Relevancia Respuesta: {metrics['answer_relevance']:.2f}")

        # 4. Calcular promedios
        avg_faithfulness = np.mean([r['metrics']['faithfulness'] for r in db_results])
        avg_context_relevance = np.mean([r['metrics']['context_relevance'] for r in db_results])
        avg_answer_relevance = np.mean([r['metrics']['answer_relevance'] for r in db_results])
        
        all_results[db_name] = {
            "avg_faithfulness": avg_faithfulness,
            "avg_context_relevance": avg_context_relevance,
            "avg_answer_relevance": avg_answer_relevance,
            "details": db_results
        }
        
        print("\n---------------------------------------------------------")
        print(f"| Resultados Promedio para {db_name}:")
        print(f"|   Fidelidad: {avg_faithfulness:.3f}")
        print(f"|   Relevancia Contexto: {avg_context_relevance:.3f}")
        print(f"|   Relevancia Respuesta: {avg_answer_relevance:.3f}")
        print("---------------------------------------------------------")

    # 5. Guardar resultados finales
    output_filename = f"rag_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Evaluación RAG completada. Resultados detallados guardados en: {output_filename}")
    print("Compara los promedios para determinar la mejor estrategia de chunking.")


if __name__ == "__main__":
    main_evaluation()