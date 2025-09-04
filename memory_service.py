"""
Servicio básico de memoria para el agente DB
"""
from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from models import AgentsMessageHistory

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
    def get_chat_history(db: Session, user_id: str, limit: int = 10) -> List[Tuple[str, str]]:
        """
        Obtiene el historial de chat en formato de tuplas (role, content)
        """
        messages = db.query(AgentsMessageHistory)\
            .filter(
                and_(
                    AgentsMessageHistory.user_id == user_id,
                    AgentsMessageHistory.agent_type == "DB"
                )
            )\
            .order_by(desc(AgentsMessageHistory.timestamp))\
            .limit(limit)\
            .all()
        
        # Revertir para orden cronológico
        messages.reverse()
        
        return [(msg.role, msg.content) for msg in messages]
    
    @staticmethod
    def cleanup_old_messages(db: Session, user_id: str, keep_count: int = 10):
        """
        Limpia mensajes antiguos
        """
        total_count = db.query(AgentsMessageHistory)\
            .filter(
                and_(
                    AgentsMessageHistory.user_id == user_id,
                    AgentsMessageHistory.agent_type == "DB"
                )
            )\
            .count()
        
        if total_count > keep_count:
            messages_to_delete = total_count - keep_count
            
            old_message_ids = db.query(AgentsMessageHistory.id)\
                .filter(
                    and_(
                        AgentsMessageHistory.user_id == user_id,
                        AgentsMessageHistory.agent_type == "DB"
                    )
                )\
                .order_by(AgentsMessageHistory.timestamp)\
                .limit(messages_to_delete)\
                .all()
            
            if old_message_ids:
                ids_to_delete = [msg_id[0] for msg_id in old_message_ids]
                
                db.query(AgentsMessageHistory)\
                    .filter(AgentsMessageHistory.id.in_(ids_to_delete))\
                    .delete(synchronize_session=False)
                
                db.commit()