from fastapi import FastAPI, Request, Depends
from sqlalchemy.orm import Session
from db_agent import db_agent_app
from langgraph.errors import GraphRecursionError
from models import get_db, DBAgentRequest, DBAgentResponse
from memory_service import MemoryService

app = FastAPI(
    title="PrevenChat API",
    description="API para agentes de chat con memoria persistente",
    version="2.0.0"
)

# RAG Agent Urls
@app.post("/api/rag/ask")
async def ask_rag(request: Request):
    data = await request.json()
    return data


@app.post("/api/rag/upload_document")
async def upload_document_rag(request: Request):
    data = await request.json()
    return data



# DB Agent Urls
@app.post("/api/db/ask", response_model=DBAgentResponse)
async def ask_db(
    request: DBAgentRequest,
    db: Session = Depends(get_db)
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
            
            # 7. Limpiar historial si es necesario
            MemoryService.cleanup_old_messages(
                db=db,
                user_id=request.user_id,
                keep_count=MAX_HISTORY_LENGTH
            )
                
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