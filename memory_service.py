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
from logger import log_exception

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
        log_exception(db)
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
        
        # Revertir para orden cronológico
        messages.reverse()
        
        return [(msg.role, msg.content) for msg in messages]
    
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