import os
import glob
import chromadb
from typing import List, Tuple, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain import hub
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

load_dotenv()

# Configuración para el agente RAG
documents_path = "documents"
persist_directory = "chroma_db"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM_MODEL_NAME = "gpt-4.1-2025-04-14"

# Variables globales para el agente
_embeddings = None
_vectorstore = None
_retriever = None
_rag_app = None
_is_initialized = False

# --- Inicialización y carga de documentos ---

class RAGAgent:
    """Agente RAG independiente para FastAPI"""
    
    def __init__(self):
        self.documents_path = documents_path
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.app = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Inicializa el agente RAG con carga automática de documentos"""
        try:
            print("🔄 Inicializando agente RAG...")
            
            # 1. Configurar embeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=OPENAI_API_KEY
            )
            
            # 2. Inicializar cliente Chroma persistente
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            
            # 3. Crear/cargar vectorstore
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name="rag_documents",
                embedding_function=self.embeddings,
            )
            
            # 4. Cargar documentos si la colección está vacía
            collection_info = self.vectorstore._collection.count()
            if collection_info == 0:
                print("📚 La colección está vacía. Cargando documentos...")
                self._load_documents()
            else:
                print(f"📚 Colección existente encontrada con {collection_info} documentos")
            
            # 5. Configurar retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # 6. Construir la aplicación RAG
            self._build_rag_graph()
            
            self.is_initialized = True
            print("✅ Agente RAG inicializado correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando agente RAG: {e}")
            return False
    
    def _load_documents(self):
        """Carga documentos desde la carpeta documents y sus subcarpetas"""
        documents = []
        
        if not os.path.exists(self.documents_path):
            print(f"⚠️ Carpeta {self.documents_path} no existe")
            return
        
        # Buscar PDFs en todas las subcarpetas
        pdf_files = glob.glob(os.path.join(self.documents_path, "**", "*.pdf"), recursive=True)
        
        if not pdf_files:
            print(f"⚠️ No se encontraron archivos PDF en {self.documents_path}")
            return
        
        print(f"📄 Encontrados {len(pdf_files)} archivos PDF")
        
        # Configurar text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        for pdf_path in pdf_files:
            try:
                print(f"📖 Procesando: {pdf_path}")
                
                # Determinar categoría basada en la estructura de carpetas
                relative_path = os.path.relpath(pdf_path, self.documents_path)
                category = os.path.dirname(relative_path) if os.path.dirname(relative_path) else "general"
                
                # Cargar y dividir el PDF
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                # Dividir en chunks
                chunks = text_splitter.split_documents(pages)
                
                # Añadir metadatos de categoría
                for chunk in chunks:
                    chunk.metadata.update({
                        "category": category,
                        "source_file": os.path.basename(pdf_path),
                        "file_path": pdf_path
                    })
                
                documents.extend(chunks)
                
            except Exception as e:
                print(f"❌ Error procesando {pdf_path}: {e}")
                continue
        
        if documents:
            print(f"📚 Añadiendo {len(documents)} documentos a la vectorstore...")
            self.vectorstore.add_documents(documents)
            print("✅ Documentos cargados exitosamente")
        else:
            print("⚠️ No se pudieron cargar documentos")
    
    def _build_rag_graph(self):
        """Construye el grafo LangGraph para el agente RAG"""
        workflow = StateGraph(GraphState)
        
        # Añadir nodos
        workflow.add_node("validate_user_input", validate_user_input)
        workflow.add_node("categorize_question", categorize_question)
        workflow.add_node("retrieve", self._create_retrieve_function())
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("transform_query", transform_query)
        workflow.add_node("simple_response", simple_response)
        
        # Añadir edges
        workflow.add_edge(START, "validate_user_input")
        workflow.add_conditional_edges(
            "validate_user_input",
            decide_after_validation,
            {
                "simple_response": "simple_response",
                "categorize_question": "categorize_question"
            }
        )
        workflow.add_edge("categorize_question", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate"
            }
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "transform_query",
                "useful": END,
                "not useful": "transform_query"
            }
        )
        workflow.add_edge("simple_response", END)
        
        # Compilar la aplicación
        self.app = workflow.compile()
    
    def _create_retrieve_function(self):
        """Crea función de recuperación personalizada que usa self.vectorstore"""
        def retrieve_with_self(state):
            return retrieve_with_vectorstore(state, self.vectorstore)
        return retrieve_with_self
    
    def reload_documents(self) -> Dict[str, Any]:
        """
        RECARGA COMPLETA de documentos - OPERACIÓN DE MANTENIMIENTO
        
        ⚠️  ADVERTENCIA: Esta operación:
        - Destruye TODA la vectorstore existente
        - Re-procesa TODOS los documentos en la carpeta 'documents'
        - Re-calcula TODOS los embeddings (costoso)
        
        📋 Usar solo cuando:
        - Cambien configuraciones (chunk_size, overlap, etc.)
        - Haya inconsistencias en la vectorstore
        - Se requiera limpieza completa del sistema
        
        💡 Para uso normal, use add_document_to_category() que es incremental
        """
        try:
            print("⚠️  INICIANDO RECARGA COMPLETA - OPERACIÓN DE MANTENIMIENTO")
            print("🔄 Esta operación destruirá y reconstruirá toda la vectorstore...")
            
            # Obtener conteo antes de la limpieza
            docs_before = self.vectorstore._collection.count()
            print(f"📊 Documentos antes de la limpieza: {docs_before}")
            
            # Limpiar la colección existente
            # ChromaDB requiere especificar qué eliminar
            try:
                # Obtener todos los IDs de la colección
                all_data = self.vectorstore._collection.get()
                if all_data['ids']:
                    self.vectorstore._collection.delete(ids=all_data['ids'])
                    print(f"🗑️ Vectorstore completamente limpiada ({len(all_data['ids'])} documentos eliminados)")
                else:
                    print("🗑️ Vectorstore ya estaba vacía")
            except Exception as delete_error:
                print(f"⚠️ Error al limpiar vectorstore: {delete_error}")
                # Si falla el delete, intentar recrear completamente la colección
                try:
                    collection_name = self.vectorstore._collection.name
                    self.vectorstore._client.delete_collection(collection_name)
                    print("🗑️ Colección eliminada, recreando...")
                    
                    # Recrear la vectorstore desde cero
                    self.vectorstore = Chroma(
                        client=self.vectorstore._client,
                        collection_name=collection_name,
                        embedding_function=self.embeddings,
                    )
                    print("✅ Vectorstore recreada correctamente")
                except Exception as recreate_error:
                    print(f"❌ Error recreando vectorstore: {recreate_error}")
                    raise recreate_error
            
            # Recargar TODOS los documentos desde la carpeta
            print("📚 Re-procesando TODOS los documentos desde la carpeta...")
            self._load_documents()
            
            # Actualizar el retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Contar documentos después de la recarga
            docs_after = self.vectorstore._collection.count()
            
            print(f"✅ RECARGA COMPLETA FINALIZADA:")
            print(f"   - Documentos antes: {docs_before}")
            print(f"   - Documentos después: {docs_after}")
            print(f"   - Operación: MANTENIMIENTO COMPLETO")
            
            return {
                "status": "success",
                "message": "Recarga completa de documentos finalizada (operación de mantenimiento)",
                "data": {
                    "documents_before": docs_before,
                    "documents_after": docs_after,
                    "documents_reprocessed": docs_after,
                    "operation_type": "full_maintenance_reload",
                    "warning": "Esta operación reprocesó TODOS los documentos y recalculó TODOS los embeddings"
                }
            }
            
        except Exception as e:
            print(f"❌ Error en recarga completa: {e}")
            return {
                "status": "error",
                "message": f"Error en recarga completa de documentos: {str(e)}",
                "data": {
                    "error_type": "full_reload_failed",
                    "operation_type": "maintenance"
                }
            }
    
    def add_document_to_category(self, category: str, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Añade un documento PDF a una categoría específica de forma INCREMENTAL
        NO destruye la vectorstore existente, solo añade el nuevo documento
        """
        try:
            print(f"📄 Iniciando upload incremental: {filename} → {category}")
            
            # Validar que es un PDF
            if not filename.lower().endswith('.pdf'):
                return {
                    "status": "error",
                    "message": "Solo se permiten archivos PDF",
                    "data": None
                }
            
            # Crear directorio de categoría si no existe
            category_path = os.path.join(self.documents_path, category)
            os.makedirs(category_path, exist_ok=True)
            print(f"📁 Directorio de categoría: {category_path}")
            
            # Definir ruta del archivo
            file_path = os.path.join(category_path, filename)
            
            # Verificar si el archivo ya existe
            if os.path.exists(file_path):
                return {
                    "status": "error",
                    "message": f"El archivo '{filename}' ya existe en la categoría '{category}'. Use el endpoint de reload si desea actualizar documentos existentes.",
                    "data": {
                        "existing_file": file_path,
                        "category": category,
                        "suggestion": "Renombre el archivo o use /api/rag/reload_documents para actualizar"
                    }
                }
            
            # Guardar el archivo físicamente
            with open(file_path, 'wb') as f:
                f.write(file_content)
            print(f"💾 Archivo guardado físicamente: {file_path}")
            
            # Obtener conteo antes del procesamiento
            docs_before = self.vectorstore._collection.count()
            
            # Procesar el documento (SOLO el nuevo)
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.document_loaders import PyPDFLoader
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            # Cargar y dividir el PDF
            print(f"📖 Procesando PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Dividir en chunks
            chunks = text_splitter.split_documents(pages)
            print(f"✂️ Documento dividido en {len(chunks)} chunks")
            
            # Añadir metadatos de categoría y timestamp
            import time
            timestamp = int(time.time())
            
            for chunk in chunks:
                chunk.metadata.update({
                    "category": category,
                    "source_file": filename,
                    "file_path": file_path,
                    "upload_timestamp": timestamp,
                    "upload_type": "incremental"
                })
            
            # AÑADIR a la vectorstore existente (NO reemplazar)
            print(f"🔄 Añadiendo {len(chunks)} chunks a vectorstore existente...")
            self.vectorstore.add_documents(chunks)
            
            # Verificar que se añadieron correctamente
            docs_after = self.vectorstore._collection.count()
            chunks_added = docs_after - docs_before
            
            print(f"✅ Upload incremental completado:")
            print(f"   - Documentos antes: {docs_before}")
            print(f"   - Documentos después: {docs_after}")
            print(f"   - Chunks añadidos: {chunks_added}")
            
            return {
                "status": "success",
                "message": f"Documento '{filename}' añadido exitosamente a la categoría '{category}' (modo incremental)",
                "data": {
                    "file_path": file_path,
                    "category": category,
                    "chunks_added": chunks_added,
                    "filename": filename,
                    "total_documents_in_vectorstore": docs_after,
                    "upload_type": "incremental",
                    "upload_timestamp": timestamp
                }
            }
            
        except Exception as e:
            print(f"❌ Error en upload incremental: {e}")
            # Si hay error, intentar limpiar el archivo guardado
            try:
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Archivo limpiado tras error: {file_path}")
            except:
                pass
                
            return {
                "status": "error",
                "message": f"Error procesando el documento: {str(e)}",
                "data": {
                    "error_type": "incremental_upload_failed",
                    "filename": filename,
                    "category": category
                }
            }

    def delete_document(self, category: str, filename: str) -> Dict[str, Any]:
        """
        Elimina un documento específico tanto de la vectorstore como del sistema de archivos
        
        Args:
            category: Categoría del documento a eliminar
            filename: Nombre del archivo a eliminar (con extensión .pdf)
            
        Returns:
            Dict con status, message y data sobre la eliminación
        """
        try:
            print(f"🗑️ Iniciando eliminación: {filename} de categoría '{category}'")
            
            # 1. Validar que es un PDF
            if not filename.lower().endswith('.pdf'):
                return {
                    "status": "error",
                    "message": "Solo se pueden eliminar archivos PDF",
                    "data": {
                        "filename": filename,
                        "category": category
                    }
                }
            
            # 2. Construir la ruta del archivo
            file_path = os.path.join(self.documents_path, category, filename)
            
            # 3. Verificar que el archivo existe físicamente
            if not os.path.exists(file_path):
                return {
                    "status": "error",
                    "message": f"El archivo '{filename}' no existe en la categoría '{category}'",
                    "data": {
                        "file_path": file_path,
                        "category": category,
                        "filename": filename
                    }
                }
            
            # 4. Obtener conteo antes de la eliminación
            docs_before = self.vectorstore._collection.count()
            
            # 5. Eliminar chunks de la vectorstore usando metadatos
            # Buscar todos los chunks que coincidan con category y source_file
            try:
                # Obtener todos los datos de la colección para filtrar
                all_data = self.vectorstore._collection.get(
                    where={
                        "$and": [
                            {"category": {"$eq": category}},
                            {"source_file": {"$eq": filename}}
                        ]
                    }
                )
                
                chunks_to_delete = all_data['ids']
                
                if chunks_to_delete:
                    print(f"🔍 Encontrados {len(chunks_to_delete)} chunks a eliminar de vectorstore")
                    self.vectorstore._collection.delete(ids=chunks_to_delete)
                    print(f"✅ {len(chunks_to_delete)} chunks eliminados de vectorstore")
                else:
                    print("⚠️ No se encontraron chunks en vectorstore para este documento")
                    
            except Exception as vectorstore_error:
                print(f"❌ Error eliminando de vectorstore: {vectorstore_error}")
                return {
                    "status": "error",
                    "message": f"Error eliminando chunks de vectorstore: {str(vectorstore_error)}",
                    "data": {
                        "error_type": "vectorstore_deletion_failed",
                        "filename": filename,
                        "category": category,
                        "file_path": file_path
                    }
                }
            
            # 6. Eliminar archivo físico
            try:
                os.remove(file_path)
                print(f"🗑️ Archivo físico eliminado: {file_path}")
            except Exception as file_error:
                print(f"❌ Error eliminando archivo físico: {file_error}")
                return {
                    "status": "error",
                    "message": f"Error eliminando archivo físico: {str(file_error)}",
                    "data": {
                        "error_type": "file_deletion_failed",
                        "filename": filename,
                        "category": category,
                        "file_path": file_path,
                        "vectorstore_chunks_deleted": len(chunks_to_delete) if 'chunks_to_delete' in locals() else 0
                    }
                }
            
            # 7. Verificar eliminación en vectorstore
            docs_after = self.vectorstore._collection.count()
            chunks_deleted = docs_before - docs_after
            
            # 8. Verificar si la carpeta de categoría está vacía y eliminarla si es necesario
            category_path = os.path.join(self.documents_path, category)
            try:
                if os.path.exists(category_path) and not os.listdir(category_path):
                    os.rmdir(category_path)
                    print(f"📁 Carpeta de categoría vacía eliminada: {category_path}")
                    category_deleted = True
                else:
                    category_deleted = False
            except Exception as category_error:
                print(f"⚠️ Error eliminando carpeta de categoría: {category_error}")
                category_deleted = False
            
            print(f"✅ Eliminación completada:")
            print(f"   - Documento: {filename}")
            print(f"   - Categoría: {category}")
            print(f"   - Chunks eliminados: {chunks_deleted}")
            print(f"   - Vectorstore antes: {docs_before}")
            print(f"   - Vectorstore después: {docs_after}")
            
            return {
                "status": "success",
                "message": f"Documento '{filename}' eliminado exitosamente de la categoría '{category}'",
                "data": {
                    "filename": filename,
                    "category": category,
                    "file_path": file_path,
                    "chunks_deleted": chunks_deleted,
                    "vectorstore_documents_before": docs_before,
                    "vectorstore_documents_after": docs_after,
                    "category_deleted": category_deleted,
                    "operation_type": "document_deletion"
                }
            }
            
        except Exception as e:
            print(f"❌ Error general eliminando documento: {e}")
            return {
                "status": "error",
                "message": f"Error eliminando documento: {str(e)}",
                "data": {
                    "error_type": "general_deletion_error",
                    "filename": filename,
                    "category": category,
                    "error_details": str(e)
                }
            }

    def ask(self, question: str, user_id: str, chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Procesa una pregunta usando el agente RAG"""
        if not self.is_initialized:
            return {
                "status": "error",
                "message": "El agente RAG no está inicializado",
                "data": None
            }
        
        try:
            # Preparar estado inicial
            initial_state = {
                "question": question,
                "generation": "",
                "documents": [],
                "attempts": 0,
                "categories": [],
                "chat_history": chat_history or [],
                "input_type": "",
                "needs_documents": True
            }
            
            # Ejecutar el agente
            result = self.app.invoke(initial_state)
            
            return {
                "status": "success",
                "message": "Consulta procesada correctamente",
                "data": {
                    "response": result.get("generation", "No se pudo generar una respuesta"),
                    "user_query": question,
                    "categories_used": result.get("categories", []),
                    "documents_found": len(result.get("documents", [])),
                    "input_type": result.get("input_type", "")
                }
            }
            
        except Exception as e:
            print(f"❌ Error procesando consulta RAG: {e}")
            return {
                "status": "error",
                "message": f"Error procesando la consulta: {str(e)}",
                "data": None
            }

# Instancia global del agente
rag_agent_instance = RAGAgent()

# Función para inicializar el agente (se llama al arrancar la aplicación)
def initialize_rag_agent() -> bool:
    """Inicializa el agente RAG global"""
    global _is_initialized
    if not _is_initialized:
        _is_initialized = rag_agent_instance.initialize()
    return _is_initialized

# Función para obtener el agente inicializado
def get_rag_agent() -> RAGAgent:
    """Obtiene la instancia del agente RAG"""
    if not rag_agent_instance.is_initialized:
        raise RuntimeError("El agente RAG no está inicializado")
    return rag_agent_instance

# --- Modelos Pydantic para LangGraph ---

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class RelevantCategories(BaseModel):
    """Identifies relevant categories for a user question."""
    categories: List[str] = Field(
        description="A list of category names relevant to the user question, selected from the provided options. Return an empty list if no category matches."
    )

class InputValidation(BaseModel):
    """Validates user input to determine the appropriate response type."""
    input_type: str = Field(
        description="Type of user input. Options: 'greeting' (saludos, despedidas), 'casual_conversation' (conversación casual), 'document_question' (pregunta que requiere búsqueda en documentos), 'unclear' (entrada ambigua)"
    )
    needs_documents: bool = Field(
        description="True if the input requires searching documents to provide a proper answer, False otherwise"
    )
    simple_response: str = Field(
        description="Direct response for greetings or casual conversation. Empty string if documents are needed."
    )

# LLM instances
llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
structured_llm_categorizer = llm.with_structured_output(RelevantCategories)
structured_llm_validator = llm.with_structured_output(InputValidation)

# Prompt para el evaluador de documentos
system = """Eres un evaluador que determina la relevancia de un documento recuperado para una pregunta del usuario.
Evalúa si el documento contiene información, datos, conceptos o análisis relacionados con la pregunta.
Si el documento contiene palabras clave o información semántica relacionada con la pregunta del usuario, 
calífícalo como relevante con "yes".
Da una puntuación binaria 'yes' o 'no' para indicar si el documento es relevante para la pregunta."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Documento recuperado: \n\n {document} \n\n Pregunta del usuario: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# Prompt para RAG
prompt = hub.pull("rlm/rag-prompt")

# Post-processing
def format_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Prompt para el evaluador de alucinaciones
system = """Eres un evaluador que determina si una respuesta generada está fundamentada en los documentos proporcionados.
Da una puntuación 'yes' si la respuesta está basada en los hechos presentados en los documentos y no incluye información externa o fabricada.
Da una puntuación 'no' si la respuesta contiene afirmaciones o datos que no están presentes en los documentos.
La respuesta 'yes' significa que la respuesta está fundamentada en los documentos proporcionados."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Conjunto de documentos: \n\n {documents} \n\n Respuesta generada: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

# Prompt para evaluador de respuestas
system = """Eres un evaluador que determina si una respuesta resuelve adecuadamente una pregunta del usuario.
Si la respuesta incluye datos relevantes relacionados con la pregunta y proporciona información útil, califícala como 'yes'.
Si la respuesta es genérica, no incluye datos específicos o no responde directamente a la pregunta, califícala como 'no'.
Da una puntuación binaria 'yes' o 'no' para indicar si la respuesta aborda adecuadamente la pregunta."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Pregunta del usuario: \n\n {question} \n\n Respuesta generada: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_answer_grader

# LLM para reescritura de preguntas
llm_rewriter = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.1, openai_api_key=OPENAI_API_KEY)

# Prompt para reescritura de preguntas
system = """Eres un especialista que reformula preguntas para mejorar la recuperación de información de documentos.
Analiza la pregunta original y reformúlala para que sea más específica y precisa, enfocándola en encontrar
información relevante dentro de documentos de cualquier tipo.
No cambies el idioma de la pregunta original.

Se amable, calido y profesional con tus respuestas."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Aquí está la pregunta inicial: \n\n {question} \n Formula una versión mejorada de la pregunta.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm_rewriter | StrOutputParser()

# Agent State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        attempts: number of attempts made
        categories: list of relevant categories identified for the question
        chat_history: a list of (role, content) tuples representing the conversation history
        input_type: type of user input (greeting, casual_conversation, document_question, unclear)
        needs_documents: boolean indicating if the input requires document search
    """

    question: str
    generation: str
    documents: List[str]
    attempts: int
    categories: List[str]
    chat_history: List[Tuple[str, str]]
    input_type: str
    needs_documents: bool

# --- Nodos del grafo ---

def validate_user_input(state):
    """
    Validates user input to determine if it requires document search or can be answered directly.
    """
    print("---VALIDATE USER INPUT---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # Prompt para la validación de entrada
    system_message = """Eres un asistente experto en analizar el tipo de entrada del usuario.
    Tu tarea es determinar si la entrada del usuario es:
    1. 'greeting' - Saludos, despedidas, agradecimientos básicos
    2. 'casual_conversation' - Conversación casual, comentarios generales, preguntas sobre ti como asistente
    3. 'document_question' - Preguntas específicas que requieren buscar información en documentos
    4. 'unclear' - Entrada ambigua o poco clara
    
    Para saludos y conversación casual, proporciona una respuesta amigable y directa.
    Para preguntas sobre documentos, indica que se necesita buscar información.
    
    Sé cálido, profesional y útil en tus respuestas."""
    
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Entrada del usuario: {question}")
    ])
    
    # Crear la cadena de validación
    validation_chain = validation_prompt | structured_llm_validator
    
    # Invocar la cadena
    try:
        result = validation_chain.invoke({"question": question})
        input_type = result.input_type
        needs_documents = result.needs_documents
        simple_response = result.simple_response
        
        print(f"---INPUT TYPE: {input_type}---")
        print(f"---NEEDS DOCUMENTS: {needs_documents}---")
        
        current_state = state.copy()
        current_state["input_type"] = input_type
        current_state["needs_documents"] = needs_documents
        
        # Si no necesita documentos, guardamos la respuesta directa
        if not needs_documents and simple_response:
            current_state["generation"] = simple_response
            print("---DIRECT RESPONSE PROVIDED---")
        
        return current_state
        
    except Exception as e:
        print(f"Error during input validation: {e}")
        # En caso de error, asumimos que es una pregunta que necesita documentos
        current_state = state.copy()
        current_state["input_type"] = "document_question"
        current_state["needs_documents"] = True
        return current_state

def categorize_question(state):
    """
    Categorizes the user question based on available document categories.
    """
    print("---CATEGORIZE QUESTION---")
    question = state["question"]
    attempts = state.get("attempts", 0)
    chat_history = state.get("chat_history", [])

    # Obtener dinámicamente las categorías (nombres de las carpetas)
    available_categories = []
   
    if os.path.exists(documents_path):
        try:
            available_categories = [d for d in os.listdir(documents_path) if os.path.isdir(os.path.join(documents_path, d))]
        except FileNotFoundError:
            print(f"Error: El directorio de documentos '{documents_path}' no fue encontrado.")
            available_categories = []
        except Exception as e:
             print(f"Error listando directorios en '{documents_path}': {e}")
             available_categories = []

    if not available_categories:
        print(f"No categories found in '{documents_path}'. Skipping categorization.")
        # Devolvemos el estado sin categorías, el nodo retrieve manejará la ausencia
        return {**state, "categories": []}

    # Prompt para la categorización
    system_message = f"""Eres un asistente experto en clasificar preguntas de usuarios según categorías de documentos. 
    Dada la pregunta del usuario y una lista de categorías disponibles, identifica TODAS las categorías que sean relevantes para responder la pregunta. 
    Las categorías disponibles son: {', '.join(available_categories)}. 
    Devuelve solo la lista de nombres de categorías relevantes. Si ninguna categoría parece relevante, devuelve una lista vacía."""
    
    categorization_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Pregunta del usuario: {question}")
    ])

    # Crear la cadena de categorización
    categorization_chain = categorization_prompt | structured_llm_categorizer

    # Invocar la cadena
    try:
        result = categorization_chain.invoke({"question": question})
        detected_categories = result.categories
        print(f"---DETECTED CATEGORIES: {detected_categories}---")
        
        valid_detected_categories = [cat for cat in detected_categories if cat in available_categories]
        if len(valid_detected_categories) != len(detected_categories):
            print("Warning: LLM returned categories not present in the available list.")
        
    except Exception as e:
        print(f"Error during categorization LLM call: {e}")
        valid_detected_categories = []

    current_state = state.copy()
    current_state["categories"] = valid_detected_categories
    return current_state

def retrieve_with_vectorstore(state, vectorstore):
    """
    Retrieve documents based on the question and identified categories.
    """
    print("---RETRIEVE DOCUMENTS---")
    question = state["question"]
    categories = state.get("categories", [])
    attempts = state.get("attempts", 0)
    chat_history = state.get("chat_history", [])

    if categories:
        print(f"--- Applying filter for categories: {categories} ---")
        try:
            retriever_with_filter = vectorstore.as_retriever(
                search_kwargs={'filter': {'category': {'$in': categories}}}
            )
            documents = retriever_with_filter.invoke(question)
        except Exception as e:
            print(f"Error applying category filter during retrieval: {e}")
            print("--- Falling back to retrieval without category filter ---")
            retriever_fallback = vectorstore.as_retriever(search_kwargs={"k": 4})
            documents = retriever_fallback.invoke(question)
    else:
        print("--- No categories specified, retrieving across all documents ---")
        retriever_general = vectorstore.as_retriever(search_kwargs={"k": 4})
        documents = retriever_general.invoke(question)

    current_state = state.copy()
    current_state["documents"] = documents
    return current_state

def generate(state):
    """
    Generate answer using the LLM, considering the chat history.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])

    # Formato de documentos
    formatted_docs = format_docs(documents)
    
    # Sistema de prompt que considera el historial
    custom_system_message_with_history = """
    Eres un asistente especializado en analizar información de documentos.
    Tu tarea es responder preguntas basándote exclusivamente en los documentos proporcionados y el historial de conversación previo si es relevante.
    Utiliza sólo los datos que aparecen en los documentos y no agregues información externa.
    Si los documentos no contienen la información necesaria para responder, y el historial tampoco ayuda, indica claramente que la información no está disponible.
    Si la pregunta actual es una continuación de la conversación anterior, usa el historial para entender el contexto.
    Mantén tus respuestas concisas y al grano.
    
    RESPONDE SIEMPRE EN FORMATO MARKDOWN.
    """
    
    # Crear la lista de mensajes para el prompt
    prompt_messages = [
        ("system", custom_system_message_with_history)
    ]

    # Añadir el historial de chat al prompt
    for role, content in chat_history:
        if role == 'user':
            prompt_messages.append(("human", content))
        elif role == 'assistant':
            prompt_messages.append(("ai", content))

    # Añadir el contexto de los documentos y la pregunta actual del usuario
    prompt_messages.append(
        ("human", f"Documentos de referencia:\\n\\n{formatted_docs}\\n\\nPregunta actual: {question}")
    )
    
    # Usar la construcción directa de la lista de mensajes:
    custom_prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    custom_chain = custom_prompt | llm | StrOutputParser()
    
    # RAG generation
    generation = custom_chain.invoke({})
    
    # Devolver todos los campos del estado
    current_state = state.copy()
    current_state["generation"] = generation
    return current_state

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    attempts = state.get("attempts", 0)
    categories = state.get("categories", [])
    chat_history = state.get("chat_history", [])

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    
    # Preservar todos los campos en el estado devuelto
    input_type = state.get("input_type", "")
    needs_documents = state.get("needs_documents", True)
    return {
        "documents": filtered_docs,
        "question": question,
        "attempts": attempts,
        "categories": categories,
        "chat_history": chat_history,
        "input_type": input_type,
        "needs_documents": needs_documents
    }

def transform_query(state):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    attempts = state.get("attempts", 0) + 1
    categories = state.get("categories", [])
    chat_history = state.get("chat_history", [])

    print(f"---ATTEMPT #{attempts}---")
    
    # Si ya superamos el número máximo de intentos, devolvemos un mensaje estándar
    if attempts > 3:
        print("---EXCEEDED MAXIMUM ATTEMPTS, RETURNING STANDARD RESPONSE---")
        better_question = "Información general disponible en los documentos"
    else:
        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
    
    print(f"---TRANSFORMED QUERY: {better_question}---")
    
    # Preservar todos los campos en el estado devuelto
    input_type = state.get("input_type", "")
    needs_documents = state.get("needs_documents", True)
    return {
        "documents": documents,
        "question": better_question,
        "attempts": attempts,
        "categories": categories,
        "chat_history": chat_history,
        "input_type": input_type,
        "needs_documents": needs_documents
    }

def simple_response(state):
    """
    Returns a simple response that was already generated during input validation.
    """
    print("---SIMPLE RESPONSE---")
    print(f"---RETURNING DIRECT RESPONSE: {state.get('generation', 'No response found')}---")
    return state

# --- Edges ---

def decide_after_validation(state):
    """
    Decides the next step after input validation.
    """
    print("---DECIDE AFTER VALIDATION---")
    needs_documents = state.get("needs_documents", True)
    input_type = state.get("input_type", "document_question")
    
    if not needs_documents and input_type in ["greeting", "casual_conversation"]:
        print("---DECISION: DIRECT RESPONSE, NO DOCUMENTS NEEDED---")
        return "simple_response"
    else:
        print("---DECISION: CONTINUE TO DOCUMENT SEARCH---")
        return "categorize_question"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    attempts = state.get("attempts", 0)

    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        if attempts >= 3:
            print("---MAXIMUM ATTEMPTS REACHED, GENERATING FINAL RESPONSE---")
            return "generate"
        return "transform_query"
    else:
        print(f"---DECISION: GENERATE WITH {len(filtered_documents)} RELEVANT DOCUMENTS---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    attempts = state.get("attempts", 0)

    # Máximo de 3 intentos para evitar bucles infinitos
    if attempts >= 3:
        print("---MAXIMUM ATTEMPTS REACHED, RETURNING BEST RESULT SO FAR---")
        return "useful"

    formatted_docs = format_docs(documents)
    
    # Si no hay documentos, generamos una respuesta que indique que no tenemos información
    if not formatted_docs:
        print("---NO RELEVANT DOCUMENTS FOUND, GENERATING FINAL RESPONSE---")
        custom_response = """
        Según los documentos analizados, no se encontró información relevante sobre este tema.
        
        Los documentos disponibles no contienen datos específicos relacionados con tu pregunta
        o no fue posible extraer información relevante en el procesamiento actual.
        """
        state["generation"] = custom_response
        return "useful"
    
    score = hallucination_grader.invoke(
        {"documents": formatted_docs, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            if attempts >= 3:
                print("---MAXIMUM ATTEMPTS REACHED, RETURNING BEST RESULT SO FAR---")
                return "useful"
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        if attempts >= 3:
            print("---MAXIMUM ATTEMPTS REACHED, RETURNING BEST RESULT SO FAR---")
            return "useful"
        return "not supported"