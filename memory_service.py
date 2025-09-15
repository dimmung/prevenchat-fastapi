"""
Servicio de memoria para los agentes DB y RAG

IMPORTANTE: Este servicio NO borra mensajes de la base de datos.
Todos los mensajes se conservan permanentemente para:
- Análisis de uso de usuarios
- Estadísticas de actividad
- Registro histórico completo

Solo se limita la cantidad de mensajes recuperados para el contexto del agente.
"""
from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from models import AgentsMessageHistory
from logger import log_warning_message

class MemoryService:
    """
    Servicio simple para manejar memoria del agente DB
    """
    
    @staticmethod
    def save_message(db: Session, user_id: str, agent_type: str, role: str, content: str):
        """
        Guarda un mensaje en el historial
        """
        db_message = AgentsMessageHistory(
            user_id=user_id,
            agent_type=agent_type,
            role=role,
            content=content
        )
        
        db.add(db_message)
        db.commit()
    
    @staticmethod
    def get_chat_history(db: Session, user_id: str, agent_type: str = "DB", limit: int = 10) -> List[Tuple[str, str]]:
        """
        Obtiene el historial de chat en formato de tuplas (role, content)
        
        NOTA IMPORTANTE: Solo recupera los últimos 'limit' mensajes para el contexto del agente.
        Todos los mensajes permanecen en la base de datos sin borrar.
        """
        try:
            # Log debug information using the logger system
            log_warning_message(
                f"Buscando historial para user_id: {user_id}, agent_type: {agent_type}, limit: {limit}",
                context="get_chat_history",
                extra_data={
                    "user_id": user_id,
                    "agent_type": agent_type,
                    "limit": limit,
                    "db_object": str(db),
                    "db_type": str(type(db)),
                    "db_bind": str(db.bind) if hasattr(db, 'bind') else "N/A",
                    "db_is_active": str(db.is_active) if hasattr(db, 'is_active') else "N/A"
                }
            )
            
            # Test database connection before query
            try:
                # Simple test query to check connection
                db.execute("SELECT 1")
                log_warning_message(
                    "Conexión a DB exitosa",
                    context="get_chat_history",
                    extra_data={"user_id": user_id, "test_query": "SELECT 1"}
                )
            except Exception as conn_error:
                log_warning_message(
                    f"Error de conexión a DB: {str(conn_error)}",
                    context="get_chat_history",
                    extra_data={
                        "user_id": user_id,
                        "connection_error": str(conn_error),
                        "error_type": type(conn_error).__name__
                    }
                )
                raise conn_error
            
            messages = db.query(AgentsMessageHistory)\
                .filter(
                    and_(
                        AgentsMessageHistory.user_id == user_id,
                        AgentsMessageHistory.agent_type == agent_type
                    )
                )\
                .order_by(desc(AgentsMessageHistory.timestamp))\
                .limit(limit)\
                .all()
            
            # Log messages found in database
            messages_info = []
            for i, msg in enumerate(messages):
                messages_info.append({
                    "index": i+1,
                    "role": msg.role,
                    "content_preview": msg.content[:100] + ('...' if len(msg.content) > 100 else ''),
                    "timestamp": str(msg.timestamp),
                    "full_content_length": len(msg.content)
                })
            
            log_warning_message(
                f"Mensajes encontrados en DB: {len(messages)}",
                context="get_chat_history",
                extra_data={
                    "user_id": user_id,
                    "agent_type": agent_type,
                    "messages_count": len(messages),
                    "messages_details": messages_info
                }
            )
            
            # Revertir para orden cronológico
            messages.reverse()
            
            log_warning_message(
                f"Mensajes después de revertir: {len(messages)}",
                context="get_chat_history",
                extra_data={
                    "user_id": user_id,
                    "agent_type": agent_type,
                    "final_messages_count": len(messages)
                }
            )
            
            return [(msg.role, msg.content) for msg in messages]
            
        except Exception as e:
            # Log the actual exception properly
            log_warning_message(
                f"Error en get_chat_history: {str(e)}",
                context="get_chat_history",
                extra_data={
                    "user_id": user_id,
                    "agent_type": agent_type,
                    "limit": limit,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "db_info": {
                        "db_object": str(db),
                        "db_type": str(type(db)),
                        "db_bind": str(db.bind) if hasattr(db, 'bind') else "N/A",
                        "db_is_active": str(db.is_active) if hasattr(db, 'is_active') else "N/A"
                    }
                }
            )
            # Re-raise the exception to be handled by the calling code
            raise e
    
    @staticmethod
    def get_total_messages_count(db: Session, user_id: str, agent_type: str = "DB") -> int:
        """
        Obtiene el conteo total de mensajes de un usuario para un agente específico
        Útil para estadísticas de uso sin borrar registros
        """
        return db.query(AgentsMessageHistory)\
            .filter(
                and_(
                    AgentsMessageHistory.user_id == user_id,
                    AgentsMessageHistory.agent_type == agent_type
                )
            )\
            .count()