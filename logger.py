"""
Sistema de logging centralizado para manejo de errores en prevenchat-fastapi.

Este módulo configura un logger con rotación automática que:
- Guarda errores en el archivo errores.log en la raíz del proyecto
- Rota el archivo cada 1000 líneas para evitar archivos demasiado grandes
- Mantiene 5 archivos de respaldo (errores.log.1, errores.log.2, etc.)
- Incluye timestamps, nivel de log y información detallada del error
"""

import logging
import logging.handlers
import os
import traceback
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class ErrorLogger:
    """Clase para manejar el logging de errores con rotación automática."""
    
    _instance: Optional['ErrorLogger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls) -> 'ErrorLogger':
        """Implementa patrón Singleton para asegurar una sola instancia del logger."""
        if cls._instance is None:
            cls._instance = super(ErrorLogger, cls).__new__(cls)
            cls._instance._setup_logger()
        return cls._instance
    
    def _setup_logger(self) -> None:
        """Configura el logger con rotación automática por líneas."""
        # Crear el logger principal
        self._logger = logging.getLogger('prevenchat_errors')
        self._logger.setLevel(logging.ERROR)
        
        # Evitar duplicar handlers si ya existen
        if self._logger.handlers:
            return
        
        # Determinar la ruta del archivo de log (raíz del proyecto)
        project_root = Path(__file__).parent
        log_file_path = project_root / "errores.log"
        
        # Crear handler con rotación por líneas (1000 líneas máximo por archivo)
        # maxBytes se aproxima asumiendo ~100 caracteres promedio por línea
        rotating_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=100000,  # Aproximadamente 1000 líneas de ~100 chars
            backupCount=5,    # Mantener 5 archivos de respaldo
            encoding='utf-8'
        )
        
        # Configurar el formato del log
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        rotating_handler.setFormatter(formatter)
        
        # Agregar el handler al logger
        self._logger.addHandler(rotating_handler)
        
        # Opcional: También mostrar errores críticos en consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.CRITICAL)
        console_formatter = logging.Formatter(
            fmt='🚨 CRITICAL ERROR | %(asctime)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self._logger.addHandler(console_handler)
    
    def log_error(self, 
                  exception: Exception, 
                  context: Optional[str] = None,
                  extra_data: Optional[Dict[str, Any]] = None,
                  level: int = logging.ERROR) -> None:
        """
        Registra un error en el archivo de log.
        
        Args:
            exception: La excepción capturada
            context: Contexto adicional sobre dónde ocurrió el error
            extra_data: Datos adicionales relevantes (diccionario)
            level: Nivel de logging (ERROR, CRITICAL, etc.)
        """
        if self._logger is None:
            self._setup_logger()
        
        # Construir el mensaje de error
        error_message = f"Exception: {type(exception).__name__}: {str(exception)}"
        
        if context:
            error_message = f"{context} | {error_message}"
        
        if extra_data:
            extra_info = " | ".join([f"{k}: {v}" for k, v in extra_data.items()])
            error_message = f"{error_message} | Extra: {extra_info}"
        
        # Obtener el traceback completo
        tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
        traceback_str = "".join(tb_lines)
        
        # Mensaje completo con traceback
        full_message = f"{error_message}\nTraceback:\n{traceback_str}"
        
        # Registrar en el log
        self._logger.log(level, full_message)
    
    def log_critical(self, 
                    exception: Exception, 
                    context: Optional[str] = None,
                    extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Registra un error crítico que también se muestra en consola."""
        self.log_error(exception, context, extra_data, level=logging.CRITICAL)
    
    def log_warning(self, 
                   message: str, 
                   context: Optional[str] = None,
                   extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Registra un warning (sin excepción)."""
        if self._logger is None:
            self._setup_logger()
        
        warning_message = message
        if context:
            warning_message = f"{context} | {warning_message}"
        
        if extra_data:
            extra_info = " | ".join([f"{k}: {v}" for k, v in extra_data.items()])
            warning_message = f"{warning_message} | Extra: {extra_info}"
        
        self._logger.warning(warning_message)


# Instancia global del logger
error_logger = ErrorLogger()

# Funciones de conveniencia para uso directo
def log_exception(exception: Exception, 
                 context: Optional[str] = None,
                 extra_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Función de conveniencia para registrar excepciones.
    
    Uso típico en un bloque try/except:
    ```python
    try:
        # código que puede fallar
        risky_operation()
    except Exception as e:
        log_exception(e, context="processing user request", 
                     extra_data={"user_id": user_id, "operation": "upload"})
        # manejar el error apropiadamente
    ```
    """
    error_logger.log_error(exception, context, extra_data)

def log_critical_exception(exception: Exception, 
                          context: Optional[str] = None,
                          extra_data: Optional[Dict[str, Any]] = None) -> None:
    """Función de conveniencia para registrar excepciones críticas."""
    error_logger.log_critical(exception, context, extra_data)

def log_warning_message(message: str, 
                       context: Optional[str] = None,
                       extra_data: Optional[Dict[str, Any]] = None) -> None:
    """Función de conveniencia para registrar warnings."""
    error_logger.log_warning(message, context, extra_data)

def get_error_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas básicas del archivo de errores.
    """
    project_root = Path(__file__).parent
    log_file_path = project_root / "errores.log"
    
    try:
        if not log_file_path.exists():
            return {
                "file_exists": False,
                "size_bytes": 0,
                "line_count": 0,
                "last_modified": None
            }
        
        stat_info = log_file_path.stat()
        
        # Contar líneas del archivo
        with open(log_file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        return {
            "file_exists": True,
            "size_bytes": stat_info.st_size,
            "line_count": line_count,
            "last_modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "path": str(log_file_path)
        }
    
    except Exception as e:
        return {
            "file_exists": False,
            "error": str(e),
            "path": str(log_file_path)
        }

# Test de funcionamiento del logger
if __name__ == "__main__":
    # Probar el logger
    print("🔧 Probando el sistema de logging...")
    
    try:
        # Simular un error para probar
        raise ValueError("Este es un error de prueba del sistema de logging")
    except Exception as e:
        log_exception(e, context="testing logger module", 
                     extra_data={"test": True, "version": "1.0"})
    
    print("✅ Logger configurado correctamente")
    print(f"📊 Estadísticas del log: {get_error_stats()}")
