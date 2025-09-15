"""
Modelos básicos para el sistema de memoria de agentes
"""
from datetime import datetime
from typing import Optional, Literal
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de base de datos
DATABASE_NAME = os.getenv("POSTGRESQL_NAME")
DATABASE_USER = os.getenv("POSTGRESQL_USER")
DATABASE_PASSWORD = os.getenv("POSTGRESQL_PASSWORD")
DATABASE_HOST = os.getenv("POSTGRESQL_HOST")
DATABASE_PORT = os.getenv("POSTGRESQL_PORT")
DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Tipos básicos
AgentType = Literal["RAG", "DB"]
MessageRole = Literal["user", "assistant"]

class AgentsMessageHistory(Base):
    """
    Modelo SQLAlchemy para la tabla existente agents_message_history
    """
    __tablename__ = "agents_message_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)  # STRING como en la tabla real
    agent_type = Column(String(10), nullable=False, index=True)
    role = Column(String(15), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

# Modelos Pydantic básicos
class DBAgentRequest(BaseModel):
    """Modelo para requests al agente DB"""
    message: str = Field(..., min_length=1, description="Pregunta del usuario")
    user_id: str = Field(..., description="ID único del usuario")  # STRING como en la tabla real
    config: dict = Field(default_factory=dict, description="Configuración adicional para el agente")

class DBAgentResponse(BaseModel):
    """Modelo de respuesta del agente DB"""
    status: Literal["success", "error"]
    message: str
    data: Optional[dict] = None

# Modelos Pydantic para el agente RAG
class RAGAgentRequest(BaseModel):
    """Modelo para requests al agente RAG"""
    message: str = Field(..., min_length=1, description="Pregunta del usuario")
    user_id: str = Field(..., description="ID único del usuario")
    config: dict = Field(default_factory=dict, description="Configuración adicional para el agente")

class RAGAgentResponse(BaseModel):
    """Modelo de respuesta del agente RAG"""
    status: Literal["success", "error"]
    message: str
    data: Optional[dict] = None

class DocumentUploadRequest(BaseModel):
    """Modelo para upload de documentos RAG"""
    category: str = Field(..., min_length=1, max_length=50, description="Categoría del documento")
    
class DocumentUploadResponse(BaseModel):
    """Modelo de respuesta para upload de documentos"""
    status: Literal["success", "error", "info"]
    message: str
    data: Optional[dict] = None

# Dependency para obtener la sesión de base de datos
def get_db():
    """
    Dependency de FastAPI para obtener una sesión de base de datos
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()