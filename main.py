from fastapi import FastAPI, Request, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from db_agent import db_agent_app
from rag_agent import initialize_rag_agent, get_rag_agent
from langgraph.errors import GraphRecursionError
from models import get_db, DBAgentRequest, DBAgentResponse, RAGAgentRequest, RAGAgentResponse, DocumentUploadResponse
from memory_service import MemoryService
from auth import auth_dependency
import re

app = FastAPI(
    title="PrevenChat API",
    description="API para agentes de chat con memoria persistente",
    version="2.0.0"
)

# Evento de inicio para inicializar el agente RAG
@app.on_event("startup")
async def startup_event():
    """Inicializa el agente RAG al arrancar la aplicación"""
    print("🚀 Iniciando aplicación...")
    try:
        success = initialize_rag_agent()
        if success:
            print("✅ Agente RAG inicializado correctamente")
        else:
            print("❌ Error inicializando agente RAG")
    except Exception as e:
        print(f"❌ Error crítico inicializando agente RAG: {e}")

# RAG Agent Urls
@app.post("/api/rag/ask", response_model=RAGAgentResponse)
async def ask_rag(
    request: RAGAgentRequest,
    db: Session = Depends(get_db),
    auth: bool = Depends(auth_dependency)
) -> RAGAgentResponse:
    """
    Endpoint para procesar consultas RAG usando el agente LangGraph con memoria
    Compatible con la tabla existente 'agents_message_history'
    
    Args:
        request: Datos de la consulta (message, user_id, config)
        db: Sesión de base de datos
        
    Returns:
        RAGAgentResponse: Respuesta del agente RAG con manejo de memoria
    """
    MAX_HISTORY_LENGTH = 5
    
    try:
        print(f"🔍 Procesando consulta RAG para usuario {request.user_id}: {request.message}")
        
        # 1. Obtener el agente RAG
        try:
            rag_agent = get_rag_agent()
        except RuntimeError as e:
            return RAGAgentResponse(
                status="error",
                message="El agente RAG no está disponible",
                data={
                    "error_type": "agent_not_initialized",
                    "user_query": request.message,
                    "error_details": str(e)
                }
            )
        
        # 2. Recuperar historial de chat para el agente RAG
        chat_history = MemoryService.get_chat_history(
            db=db,
            user_id=request.user_id,
            agent_type="RAG",
            limit=MAX_HISTORY_LENGTH
        )
        
        print(f"📝 Historial RAG ({len(chat_history)} mensajes): {len(chat_history)} históricos")
        
        # 3. Procesar la consulta con el agente RAG
        result = rag_agent.ask(
            question=request.message,
            user_id=request.user_id,
            chat_history=chat_history
        )
        
        # 4. Verificar el resultado del agente
        if result["status"] == "error":
            return RAGAgentResponse(
                status="error",
                message=result["message"],
                data=result["data"]
            )
        
        # 5. Extraer la respuesta del agente
        final_response = result["data"]["response"]
        
        print(f"✅ Respuesta RAG generada para usuario {request.user_id}: {final_response[:100]}...")
        
        # 6. Guardar la pregunta y respuesta en el historial
        try:
            # Guardar pregunta del usuario
            MemoryService.save_message(
                db=db,
                user_id=request.user_id,
                agent_type="RAG",
                role="user",
                content=request.message
            )
            
            # Guardar respuesta del asistente
            MemoryService.save_message(
                db=db,
                user_id=request.user_id,
                agent_type="RAG",
                role="assistant",
                content=final_response
            )
            
            # 7. Registrar estadísticas de uso (sin borrar registros)
            total_rag_messages = MemoryService.get_total_messages_count(
                db=db,
                user_id=request.user_id,
                agent_type="RAG"
            )
            print(f"📊 Usuario {request.user_id} - Total mensajes RAG: {total_rag_messages}")
                
        except Exception as memory_error:
            print(f"⚠️ Error al guardar en memoria RAG: {memory_error}")
            # No fallar la respuesta si hay error de memoria
        
        return RAGAgentResponse(
            status="success", 
            message="Consulta RAG procesada correctamente",
            data={
                "response": final_response,
                "user_query": request.message,
                "categories_used": result["data"].get("categories_used", []),
                "documents_found": result["data"].get("documents_found", 0),
                "input_type": result["data"].get("input_type", ""),
                "history_length": len(chat_history)
            }
        )
        
    except Exception as e:
        print(f"❌ Error general procesando consulta RAG: {e}")
        return RAGAgentResponse(
            status="error",
            message=f"Error procesando la consulta RAG: {str(e)}",
            data={
                "error_type": "general_error",
                "user_query": request.message,
                "error_details": str(e)
            }
        )


@app.post("/api/rag/upload_document", response_model=DocumentUploadResponse)
async def upload_document_rag(
    category: str = Form(..., description="Categoría del documento"),
    file: UploadFile = File(..., description="Archivo PDF a subir"),
    auth: bool = Depends(auth_dependency)
) -> DocumentUploadResponse:
    """
    📄 UPLOAD INCREMENTAL - Añade UN documento nuevo sin afectar los existentes
    
    🚀 VENTAJAS:
    - ⚡ Rápido: Solo procesa el documento nuevo
    - 💰 Económico: Solo genera embeddings del archivo nuevo
    - 🔄 Eficiente: Ideal para uso diario con 200+ documentos
    - 🎯 Seguro: No afecta documentos existentes
    
    Args:
        category: Categoría donde se guardará el documento (se crea si no existe)
        file: Archivo PDF a procesar
        
    Returns:
        DocumentUploadResponse: Resultado del upload incremental
        
    📋 Uso recomendado: Para añadir documentos nuevos día a día
    """
    try:
        print(f"📄 Recibiendo upload: {file.filename} en categoría '{category}'")
        
        # 1. Validaciones básicas
        if not file.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó un archivo")
        
        if not file.filename.lower().endswith('.pdf'):
            return DocumentUploadResponse(
                status="error",
                message="Solo se permiten archivos PDF",
                data={
                    "file_type": file.content_type,
                    "filename": file.filename
                }
            )
        
        # 2. Validar nombre de categoría
        if not re.match(r'^[a-zA-Z0-9_-]+$', category):
            return DocumentUploadResponse(
                status="error",
                message="El nombre de la categoría solo puede contener letras, números, guiones y guiones bajos",
                data={
                    "category": category,
                    "pattern": "a-zA-Z0-9_-"
                }
            )
        
        # 3. Obtener el agente RAG
        try:
            rag_agent = get_rag_agent()
        except RuntimeError as e:
            return DocumentUploadResponse(
                status="error",
                message="El agente RAG no está disponible",
                data={
                    "error_type": "agent_not_initialized",
                    "error_details": str(e)
                }
            )
        
        # 4. Leer el contenido del archivo
        file_content = await file.read()
        
        if len(file_content) == 0:
            return DocumentUploadResponse(
                status="error",
                message="El archivo está vacío",
                data={
                    "filename": file.filename,
                    "size": len(file_content)
                }
            )
        
        # 5. Verificar tamaño del archivo (máximo 20MB)
        max_size = 20 * 1024 * 1024  # 20MB
        if len(file_content) > max_size:
            return DocumentUploadResponse(
                status="error",
                message=f"El archivo es demasiado grande. Máximo permitido: {max_size // (1024*1024)}MB",
                data={
                    "filename": file.filename,
                    "size": len(file_content),
                    "max_size": max_size
                }
            )
        
        # 6. Procesar el documento con el agente RAG
        result = rag_agent.add_document_to_category(
            category=category,
            file_content=file_content,
            filename=file.filename
        )
        
        # 7. Mapear la respuesta del agente al modelo de respuesta
        return DocumentUploadResponse(
            status=result["status"],
            message=result["message"],
            data=result["data"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error general en upload de documento: {e}")
        return DocumentUploadResponse(
            status="error",
            message=f"Error procesando el archivo: {str(e)}",
            data={
                "error_type": "general_error",
                "filename": file.filename if file and file.filename else "unknown",
                "category": category
            }
        )


@app.post("/api/rag/reload_documents", response_model=DocumentUploadResponse)
async def reload_documents_rag(
    auth: bool = Depends(auth_dependency)
) -> DocumentUploadResponse:
    """
    🔧 RECARGA COMPLETA - Operación de MANTENIMIENTO (usar con precaución)
    
    ⚠️  ADVERTENCIA:
    - 🗑️ Destruye TODA la vectorstore existente
    - 🔄 Re-procesa TODOS los documentos (200+ docs)
    - 💸 Re-calcula TODOS los embeddings (costoso)
    - ⏱️ Operación lenta (minutos con muchos documentos)
    
    📋 USAR SOLO CUANDO:
    - Cambien configuraciones del sistema (chunk_size, etc.)
    - Haya inconsistencias en la vectorstore
    - Se requiera limpieza completa
    - Documentos existentes estén corruptos
    
    💡 Para uso normal: Use /api/rag/upload_document (incremental)
    
    Returns:
        DocumentUploadResponse: Resultado de la recarga completa
    """
    try:
        print("🔄 Solicitud de recarga de documentos recibida")
        
        # 1. Obtener el agente RAG
        try:
            rag_agent = get_rag_agent()
        except RuntimeError as e:
            return DocumentUploadResponse(
                status="error",
                message="El agente RAG no está disponible",
                data={
                    "error_type": "agent_not_initialized",
                    "error_details": str(e)
                }
            )
        
        # 2. Recargar documentos
        result = rag_agent.reload_documents()
        
        # 3. Mapear la respuesta
        return DocumentUploadResponse(
            status=result["status"],
            message=result["message"],
            data=result["data"]
        )
        
    except Exception as e:
        print(f"❌ Error general en recarga de documentos: {e}")
        return DocumentUploadResponse(
            status="error",
            message=f"Error recargando documentos: {str(e)}",
            data={
                "error_type": "general_error",
                "error_details": str(e)
            }
        )


@app.get("/api/rag/stats", response_model=DocumentUploadResponse)
async def get_rag_stats(
    auth: bool = Depends(auth_dependency)
) -> DocumentUploadResponse:
    """
    📊 ESTADÍSTICAS del agente RAG - Información del estado actual
    
    🔍 Información proporcionada:
    - Total de documentos en vectorstore
    - Categorías disponibles
    - Archivos por categoría
    - Estado del agente
    
    Returns:
        DocumentUploadResponse: Estadísticas actuales del sistema
    """
    try:
        # 1. Obtener el agente RAG
        try:
            rag_agent = get_rag_agent()
        except RuntimeError as e:
            return DocumentUploadResponse(
                status="error",
                message="El agente RAG no está disponible",
                data={
                    "error_type": "agent_not_initialized",
                    "error_details": str(e)
                }
            )
        
        # 2. Obtener estadísticas de la vectorstore
        total_documents = rag_agent.vectorstore._collection.count()
        
        # 3. Explorar categorías y archivos
        import os
        documents_info = {}
        categories = []
        total_files = 0
        
        if os.path.exists("documents"):
            for item in os.listdir("documents"):
                item_path = os.path.join("documents", item)
                if os.path.isdir(item_path):
                    categories.append(item)
                    pdf_files = [f for f in os.listdir(item_path) if f.lower().endswith('.pdf')]
                    documents_info[item] = {
                        "files_count": len(pdf_files),
                        "files": pdf_files
                    }
                    total_files += len(pdf_files)
        
        return DocumentUploadResponse(
            status="success",
            message="Estadísticas del agente RAG obtenidas correctamente",
            data={
                "vectorstore_documents": total_documents,
                "total_categories": len(categories),
                "categories": categories,
                "total_pdf_files": total_files,
                "documents_by_category": documents_info,
                "agent_status": "initialized" if rag_agent.is_initialized else "not_initialized",
                "documents_path": "documents/",
                "vectorstore_type": "chroma_persistent"
            }
        )
        
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas RAG: {e}")
        return DocumentUploadResponse(
            status="error",
            message=f"Error obteniendo estadísticas: {str(e)}",
            data={
                "error_type": "stats_error",
                "error_details": str(e)
            }
        )


# DB Agent Urls
@app.post("/api/db/ask", response_model=DBAgentResponse)
async def ask_db(
    request: DBAgentRequest,
    db: Session = Depends(get_db),
    auth: bool = Depends(auth_dependency)
) -> DBAgentResponse:
    """
    Endpoint para procesar consultas de base de datos usando el agente LangGraph con memoria
    Compatible con la tabla existente 'agents_message_history'
    
    Args:
        request: Datos de la consulta (message, user_id, config)
        db: Sesión de base de datos
        
    Returns:
        DBAgentResponse: Respuesta del agente con manejo de memoria
    """
    MAX_HISTORY_LENGTH = 5
    
    try:
        print(f"🔍 Procesando consulta para usuario {request.user_id}: {request.message}")
        
        # 1. Recuperar historial de chat para el agente DB
        chat_history = MemoryService.get_chat_history(
            db=db,
            user_id=request.user_id,
            agent_type="DB",
            limit=MAX_HISTORY_LENGTH
        )
        
        # 2. Preparar mensajes para el agente (historial + nueva pregunta)
        current_question = ("user", request.message)
        messages_for_agent = chat_history + [current_question]
        
        print(f"📝 Mensajes para DB Agent ({len(messages_for_agent)}): {len(chat_history)} históricos + 1 actual")
        
        # 3. Configuración para el agente con límites conservadores
        recursion_limit = 7
        config = request.config.copy()
        
        print(f"⚙️ Límite de recursión establecido: {recursion_limit}")
        
        # 4. Invocar el agente LangGraph con manejo de errores de recursión
        try:
            result = db_agent_app.invoke(
                {"messages": messages_for_agent},
                recursion_limit=recursion_limit,
                config=config
            )
        except GraphRecursionError as e:
            print(f"❌ Error de recursión detectado: {e}")
            return DBAgentResponse(
                status="error",
                message="El agente ha alcanzado el límite máximo de iteraciones. Por favor, intenta reformular tu pregunta de manera más específica.",
                data={
                    "error_type": "recursion_limit",
                    "user_query": request.message,
                    "suggestion": "Intenta hacer una pregunta más específica."
                }
            )
        
        # 5. Extraer la respuesta final del agente
        final_response = "No se pudo obtener una respuesta del agente DB."
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                final_response = last_message.content
            else:
                final_response = str(last_message)
        
        print(f"✅ Respuesta generada para usuario {request.user_id}: {final_response[:100]}...")
        
        # 6. Guardar la pregunta y respuesta en el historial
        try:
            # Guardar pregunta del usuario
            MemoryService.save_message(
                db=db,
                user_id=request.user_id,
                agent_type="DB",
                role="user",
                content=request.message
            )
            
            # Guardar respuesta del asistente
            MemoryService.save_message(
                db=db,
                user_id=request.user_id,
                agent_type="DB",
                role="assistant",
                content=final_response
            )
            
            # 7. Registrar estadísticas de uso (sin borrar registros)
            total_db_messages = MemoryService.get_total_messages_count(
                db=db,
                user_id=request.user_id,
                agent_type="DB"
            )
            print(f"📊 Usuario {request.user_id} - Total mensajes DB: {total_db_messages}")
                
        except Exception as memory_error:
            print(f"⚠️ Error al guardar en memoria: {memory_error}")
            # No fallar la respuesta si hay error de memoria
        
        return DBAgentResponse(
            status="success", 
            message="Consulta procesada correctamente",
            data={
                "response": final_response,
                "user_query": request.message,
                "message_count": len(result.get("messages", [])),
                "history_length": len(chat_history)
            }
        )
        
    except Exception as e:
        print(f"❌ Error general procesando consulta: {e}")
        return DBAgentResponse(
            status="error",
            message=f"Error procesando la consulta: {str(e)}",
            data={
                "error_type": "general_error",
                "user_query": request.message,
                "error_details": str(e)
            }
        )