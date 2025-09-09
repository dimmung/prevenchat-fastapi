import hashlib
import time
import os
from functools import wraps
from fastapi import HTTPException, Request
from typing import Callable, Any
from dotenv import load_dotenv
from logger import log_exception, log_critical_exception, log_warning_message

load_dotenv()

def generate_auth_token(secret: str, timestamp: int) -> str:
    """
    Generate MD5 token from secret and timestamp.
    
    Args:
        secret (str): The secret key
        timestamp (int): Unix timestamp
    
    Returns:
        str: MD5 hash of secret + timestamp
    """
    combined = f"{secret}{timestamp}"
    return hashlib.md5(combined.encode()).hexdigest()


async def validate_token_auth(request: Request) -> bool:
    """
    Validate MD5 token authentication based on timestamp and secret.
    Expected headers:
    - X-Timestamp: Unix timestamp
    - X-Token: MD5 hash of (CHAT_SECRET + timestamp)
    
    Validates:
    1. Token matches expected MD5 hash
    2. Timestamp is not older than 5 seconds
    3. Timestamp is not from the future (more than 5 seconds)
    
    Args:
        request: FastAPI Request object
        
    Returns:
        bool: True if authentication is valid
        
    Raises:
        HTTPException: If authentication fails
    """
    # Get headers
    timestamp_header = request.headers.get('X-Timestamp')
    token_header = request.headers.get('X-Token')
    
    if not timestamp_header or not token_header:
        raise HTTPException(
            status_code=401,
            detail={
                'error': 'Unauthorized',
                'message': 'X-Timestamp and X-Token headers are required'
            }
        )
    
    try:
        # Convert timestamp to integer
        received_timestamp = int(timestamp_header)
    except ValueError as e:
        log_warning_message("Invalid timestamp format received in authentication", 
                           context="validate_token_auth - timestamp parsing", 
                           extra_data={"timestamp_header": timestamp_header, "token_header_present": bool(token_header)})
        raise HTTPException(
            status_code=401,
            detail={
                'error': 'Unauthorized',
                'message': 'Invalid timestamp format'
            }
        )
    
    # Get current timestamp
    current_timestamp = int(time.time())
    
    # Check if timestamp is not older than 5 seconds
    if current_timestamp - received_timestamp > 5:
        raise HTTPException(
            status_code=401,
            detail={
                'error': 'Unauthorized',
                'message': 'Token expired (timestamp too old)'
            }
        )
    
    # Check if timestamp is not from the future (more than 5 seconds)
    if received_timestamp - current_timestamp > 5:
        raise HTTPException(
            status_code=401,
            detail={
                'error': 'Unauthorized',
                'message': 'Invalid timestamp (from future)'
            }
        )
    
    # Get secret from environment variables
    chat_secret = os.getenv('CHAT_SECRET', '')
    if not chat_secret:
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Server Error',
                'message': 'CHAT_SECRET not configured'
            }
        )
    
    # Generate expected token
    expected_token = generate_auth_token(chat_secret, received_timestamp)
    
    # DEBUG: Log para troubleshooting (remover en producción)
    print(f"🔍 DEBUG AUTH:")
    print(f"   - chat_secret: '{chat_secret}' (type: {type(chat_secret)})")
    print(f"   - timestamp: {received_timestamp} (type: {type(received_timestamp)})")
    print(f"   - combined: '{chat_secret}{received_timestamp}'")
    print(f"   - expected_token: {expected_token}")
    print(f"   - received_token: {token_header}")
    print(f"   - tokens_match: {token_header == expected_token}")
    
    # Validate token
    if token_header != expected_token:
        raise HTTPException(
            status_code=401,
            detail={
                'error': 'Unauthorized',
                'message': 'Invalid token'
            }
        )
    
    return True


def require_auth(func: Callable) -> Callable:
    """
    Decorator for FastAPI endpoints that require token authentication.
    
    Usage:
        @app.post("/protected-endpoint")
        @require_auth
        async def protected_endpoint(request: Request, ...):
            # Your endpoint logic here
            pass
    
    Args:
        func: The FastAPI endpoint function to protect
        
    Returns:
        Wrapped function with authentication validation
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Find the Request object in the arguments
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        # If Request not found in args, check kwargs
        if request is None:
            request = kwargs.get('request')
        
        if request is None:
            raise HTTPException(
                status_code=500,
                detail={
                    'error': 'Server Error',
                    'message': 'Request object not found in endpoint parameters'
                }
            )
        
        # Validate authentication
        await validate_token_auth(request)
        
        # If authentication passes, call the original function
        return await func(*args, **kwargs)
    
    return wrapper


class AuthDependency:
    """
    FastAPI Dependency class for token authentication.
    
    Usage:
        auth_dep = AuthDependency()
        
        @app.post("/protected-endpoint")
        async def protected_endpoint(request: Request, auth: bool = Depends(auth_dep)):
            # Your endpoint logic here
            pass
    """
    
    async def __call__(self, request: Request) -> bool:
        """
        Dependency function that validates authentication.
        
        Args:
            request: FastAPI Request object
            
        Returns:
            bool: True if authentication is valid
            
        Raises:
            HTTPException: If authentication fails
        """
        return await validate_token_auth(request)


# Create a global instance for use as dependency
auth_dependency = AuthDependency()
