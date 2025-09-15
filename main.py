from fastapi import FastAPI, Request, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from db_agent import db_agent_app
from rag_agent import initialize_rag_agent, get_rag_agent
from langgraph.errors import GraphRecursionError
from models import get_db, DBAgentRequest, DBAgentResponse, RAGAgentRequest, RAGAgentResponse, DocumentUploadResponse
from memory_service import MemoryService
from auth import auth_dependency
from logger import log_exception, log_critical_exception, log_warning_message, log_info_message, log_success_message, log_debug_message
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
    log_info_message("Starting application...", context="startup_event")
    try:
        success = initialize_rag_agent()
        if success:
            log_success_message("RAG agent initialized correctly", context="startup_event")
        else:
            log_warning_message("Error initializing RAG agent", context="startup_event")
    except Exception as e:
        log_critical_exception(e, context="startup_event - initializing RAG agent", 
                             extra_data={"event": "app_startup"})
        log_warning_message(f"Critical error initializing RAG agent: {e}", context="startup_event")

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
        log_debug_message(f"Processing RAG query for user {request.user_id}: {request.message}", 
                         context="ask_rag", extra_data={"user_id": request.user_id})
        
        # 1. Obtener el agente RAG
        try:
            rag_agent = get_rag_agent()
        except RuntimeError as e:
            log_exception(e, context="ask_rag - getting RAG agent", 
                         extra_data={"user_id": request.user_id, "message": request.message[:100]})
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
        
        log_debug_message(f"RAG history ({len(chat_history)} messages): {len(chat_history)} historical", 
                         context="ask_rag", extra_data={"user_id": request.user_id, "history_length": len(chat_history)})
        
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
        
        log_success_message(f"RAG response generated for user {request.user_id}: {final_response[:100]}...", 
                           context="ask_rag", extra_data={"user_id": request.user_id})
        
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
            log_info_message(f"User {request.user_id} - Total RAG messages: {total_rag_messages}", 
                           context="ask_rag", extra_data={"user_id": request.user_id, "total_messages": total_rag_messages})
                
        except Exception as memory_error:
            log_exception(memory_error, context="ask_rag - saving to memory", 
                         extra_data={"user_id": request.user_id, "operation": "save_message"})
            log_warning_message(f"Error saving to RAG memory: {memory_error}", 
                              context="ask_rag", extra_data={"user_id": request.user_id})
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
        log_exception(e, context="ask_rag - general error", 
                     extra_data={"user_id": request.user_id, "message": request.message[:100]})
        log_warning_message(f"General error processing RAG query: {e}", 
                          context="ask_rag", extra_data={"user_id": request.user_id})
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
        log_info_message(f"Receiving upload: {file.filename} in category '{category}'", 
                        context="upload_document_rag", extra_data={"filename": file.filename, "category": category})
        
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
            log_exception(e, context="upload_document_rag - getting RAG agent", 
                         extra_data={"category": category, "filename": file.filename})
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
        log_exception(e, context="upload_document_rag - general error", 
                     extra_data={"category": category, "filename": file.filename if file and file.filename else "unknown"})
        log_warning_message(f"General error in document upload: {e}", 
                          context="upload_document_rag", extra_data={"category": category})
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
        log_info_message("Document reload request received", context="reload_documents_rag")
        
        # 1. Obtener el agente RAG
        try:
            rag_agent = get_rag_agent()
        except RuntimeError as e:
            log_exception(e, context="reload_documents_rag - getting RAG agent", 
                         extra_data={"operation": "reload_documents"})
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
        log_exception(e, context="reload_documents_rag - general error", 
                     extra_data={"operation": "reload_documents"})
        log_warning_message(f"General error in document reload: {e}", context="reload_documents_rag")
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
            log_exception(e, context="get_rag_stats - getting RAG agent", 
                         extra_data={"operation": "get_stats"})
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
        log_exception(e, context="get_rag_stats - general error", 
                     extra_data={"operation": "get_stats"})
        log_warning_message(f"Error getting RAG stats: {e}", context="get_rag_stats")
        return DocumentUploadResponse(
            status="error",
            message=f"Error obteniendo estadísticas: {str(e)}",
            data={
                "error_type": "stats_error",
                "error_details": str(e)
            }
        )


@app.delete("/api/rag/delete_document", response_model=DocumentUploadResponse)
async def delete_document_rag(
    category: str = Form(..., description="Categoría del documento a eliminar"),
    filename: str = Form(..., description="Nombre del archivo a eliminar (con extensión .pdf)"),
    auth: bool = Depends(auth_dependency)
) -> DocumentUploadResponse:
    """
    🗑️ ELIMINACIÓN DE DOCUMENTO - Elimina un documento específico
    
    ⚠️  OPERACIÓN DESTRUCTIVA:
    - 🗑️ Elimina el archivo físico de la carpeta documents/{category}
    - 🔥 Elimina TODOS los chunks del documento de la vectorstore
    - 📁 Elimina la carpeta de categoría si queda vacía
    - ❌ Esta operación NO se puede deshacer
    
    🎯 PARÁMETROS:
    - category: Nombre de la categoría donde está el documento
    - filename: Nombre exacto del archivo PDF (incluir .pdf)
    
    📋 CASOS DE USO:
    - Eliminar documentos obsoletos o incorrectos
    - Limpiar documentos duplicados
    - Mantenimiento de la base de documentos
    
    💡 RECOMENDACIÓN: Verificar con /api/rag/stats antes de eliminar
    
    Args:
        category: Categoría del documento a eliminar
        filename: Nombre del archivo PDF a eliminar
        
    Returns:
        DocumentUploadResponse: Resultado de la eliminación
    """
    try:
        log_info_message(f"Delete request: {filename} from category '{category}'", 
                        context="delete_document_rag", extra_data={"filename": filename, "category": category})
        
        # 1. Validaciones básicas
        if not filename:
            raise HTTPException(status_code=400, detail="No se proporcionó un nombre de archivo")
        
        if not category:
            raise HTTPException(status_code=400, detail="No se proporcionó una categoría")
        
        # 2. Validar que es un PDF
        if not filename.lower().endswith('.pdf'):
            return DocumentUploadResponse(
                status="error",
                message="Solo se pueden eliminar archivos PDF",
                data={
                    "filename": filename,
                    "category": category,
                    "file_type": "not_pdf"
                }
            )
        
        # 3. Validar nombre de categoría
        if not re.match(r'^[a-zA-Z0-9_-]+$', category):
            return DocumentUploadResponse(
                status="error",
                message="El nombre de la categoría solo puede contener letras, números, guiones y guiones bajos",
                data={
                    "category": category,
                    "filename": filename,
                    "pattern": "a-zA-Z0-9_-"
                }
            )
        
        # 4. Obtener el agente RAG
        try:
            rag_agent = get_rag_agent()
        except RuntimeError as e:
            log_exception(e, context="delete_document_rag - getting RAG agent", 
                         extra_data={"category": category, "filename": filename})
            return DocumentUploadResponse(
                status="error",
                message="El agente RAG no está disponible",
                data={
                    "error_type": "agent_not_initialized",
                    "filename": filename,
                    "category": category,
                    "error_details": str(e)
                }
            )
        
        # 5. Eliminar el documento usando el agente RAG
        result = rag_agent.delete_document(
            category=category,
            filename=filename
        )
        
        # 6. Mapear la respuesta del agente al modelo de respuesta
        return DocumentUploadResponse(
            status=result["status"],
            message=result["message"],
            data=result["data"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_exception(e, context="delete_document_rag - general error", 
                     extra_data={"category": category if 'category' in locals() else "unknown", 
                                "filename": filename if 'filename' in locals() else "unknown"})
        log_warning_message(f"General error in document deletion: {e}", 
                          context="delete_document_rag", extra_data={"category": category if 'category' in locals() else "unknown"})
        return DocumentUploadResponse(
            status="error",
            message=f"Error eliminando el documento: {str(e)}",
            data={
                "error_type": "general_error",
                "filename": filename if 'filename' in locals() else "unknown",
                "category": category if 'category' in locals() else "unknown",
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
        log_debug_message(f"Processing query for user {request.user_id}: {request.message}", 
                         context="ask_db", extra_data={"user_id": request.user_id})
        
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
        
        log_debug_message(f"Messages for DB Agent ({len(messages_for_agent)}): {len(chat_history)} historical + 1 current", 
                         context="ask_db", extra_data={"user_id": request.user_id, "total_messages": len(messages_for_agent)})
        
        # 3. Configuración para el agente con límites conservadores
        recursion_limit = 10
        config = request.config.copy()
        
        log_debug_message(f"Recursion limit set: {recursion_limit}", 
                         context="ask_db", extra_data={"user_id": request.user_id, "recursion_limit": recursion_limit})
        
        # 4. Invocar el agente LangGraph con manejo de errores de recursión
        try:
            # Add recursion_limit to config instead of as a separate parameter
            config_with_recursion = {**config, "recursion_limit": recursion_limit}
            result = db_agent_app.invoke(
                {"messages": messages_for_agent},
                config=config_with_recursion
            )
        except GraphRecursionError as e:
            log_exception(e, context="ask_db - recursion limit reached", 
                         extra_data={"user_id": request.user_id, "message": request.message[:100], "recursion_limit": recursion_limit})
            log_warning_message(f"Recursion error detected: {e}", 
                              context="ask_db", extra_data={"user_id": request.user_id})
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
        
        log_success_message(f"Response generated for user {request.user_id}: {final_response[:100]}...", 
                           context="ask_db", extra_data={"user_id": request.user_id})
        
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
            log_info_message(f"User {request.user_id} - Total DB messages: {total_db_messages}", 
                           context="ask_db", extra_data={"user_id": request.user_id, "total_messages": total_db_messages})
                
        except Exception as memory_error:
            log_exception(memory_error, context="ask_db - saving to memory", 
                         extra_data={"user_id": request.user_id, "operation": "save_message"})
            log_warning_message(f"Error saving to memory: {memory_error}", 
                              context="ask_db", extra_data={"user_id": request.user_id})
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
        log_exception(e, context="ask_db - general error", 
                     extra_data={"user_id": request.user_id, "message": request.message[:100]})
        log_warning_message(f"General error processing query: {e}", 
                          context="ask_db", extra_data={"user_id": request.user_id})
        return DBAgentResponse(
            status="error",
            message=f"Error procesando la consulta: {str(e)}",
            data={
                "error_type": "general_error",
                "user_query": request.message,
                "error_details": str(e)
            }
        )