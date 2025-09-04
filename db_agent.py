from typing import Annotated, Literal
from typing_extensions import TypedDict
from typing import Any
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from datetime import datetime

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM_MODEL_NAME = "gpt-4.1-2025-04-14"

DB_URI = os.getenv('DATABASE_URI')
db = SQLDatabase.from_uri(DB_URI)

# LLM para el agente de DB
llm_db = ChatOpenAI(model=LLM_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0)

# --- Definición de Herramientas y Lógica del Agente (sin cambios en la lógica interna) ---

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


toolkit = SQLDatabaseToolkit(db=db, llm=llm_db)
tools = toolkit.get_tools()
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")


@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    try:
        result = db.run_no_throw(query)
        
        if result is None:
            return "Error: Query execution failed. This might be due to syntax error, missing table, or invalid column names. Please check the table schema and rewrite the query."
        elif result == "":
            return "Query executed successfully but returned empty result. This might mean no data matches the criteria or the table is empty."
        elif isinstance(result, str) and result.strip() == "":
            return "Query executed successfully but returned empty result. This might mean no data matches the criteria or the table is empty."
        else:
            return str(result)
            
    except Exception as e:
        return f"Error executing query: {str(e)}. Please check your SQL syntax and try again."


# --- Query Check --- 
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""
query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), 
     # El placeholder espera una lista de mensajes. Le pasaremos la consulta SQL como un mensaje.
     MessagesPlaceholder(variable_name="query_messages_for_check") 
    ]
)
# LLM específico para verificar la consulta y que DEBE llamar a db_query_tool
llm_bound_for_query_check = llm_db.bind_tools([db_query_tool], tool_choice="required")
# Cadena completa para la verificación de la consulta
query_check_chain = query_check_prompt | llm_bound_for_query_check

# --- State --- 
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# --- Workflow --- 
workflow = StateGraph(State)

# --- Nodos del Grafo ---
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_list_tables", # ID de ejemplo
                    }
                ],
            )
        ]
    }

# Nodo para que el LLM principal genere la consulta SQL.
query_gen_system_prompt_text = f"""Eres un experto en SQL con gran atención al detalle.
IMPORTANTE: Debes generar una consulta SQL para responder a la pregunta.

Considera el esquema de la tabla y la pregunta del usuario para generar una consulta SQL sintácticamente correcta.
No intentes responder directamente sin generar una consulta.
La consulta generada será verificada y ejecutada en un paso posterior.

### ESTRUCTURA DE LA BASE DE DATOS:
- Esta base de datos tiene 3 tablas: delitos, eventos y recomendaciones.
- La tabla delitos contiene información de delitos ocurridos según comuna, cuadrantes, unidades policiales, etc.
- La tabla eventos tiene información de eventos de orden público, como marchas, manifestaciones, etc.
- La tabla recomendaciones tiene información de recomendaciones policiales de seguridad.

### CLASIFICACIÓN DE DELITOS EN TABLA 'delitos' (columna 'nombre_delito'):

**Delitos NO relacionados con robos:**
- 'HOMICIDIOS', 'VIOLACIONES', 'LESIONES', 'HURTOS'

**ROBOS - CLASIFICACIÓN DETALLADA:**

**Robos con Violencia (4 tipos):**
- 'ROBO CON VIOLENCIA'
- 'ROBO CON INTIMIDACIÓN' 
- 'ROBO POR SORPRESA'
- 'ROBO VIOLENTO DE VEHÍCULO'

**Robos con Fuerza (5 tipos):**
- 'ROBO DE VEHÍCULO'
- 'ROBO DE OBJETO DE O DESDE VEHÍCULO'
- 'ROBO EN LUGAR HABITADO'
- 'ROBO EN LUGAR NO HABITADO'
- 'OTROS ROBOS CON FUERZA'

### CRITERIOS PARA CONSULTAS SOBRE ROBOS:
**CRÍTICO - Aplica estas reglas según la pregunta:**

1. **"robos" o "robo" (término general)** → Incluir TODOS los 9 tipos de robos (violencia + fuerza)
2. **"robos con violencia" o "robos violentos"** → Incluir SOLO los 4 tipos de robos con violencia
3. **"robos con fuerza"** → Incluir SOLO los 5 tipos de robos con fuerza
4. **Tipos específicos mencionados** → Incluir solo ese tipo específico

### EJEMPLOS DE CONSULTAS:
- "¿Cuántos robos hubo?" → WHERE nombre_delito IN ('ROBO CON VIOLENCIA', 'ROBO CON INTIMIDACIÓN', 'ROBO POR SORPRESA', 'ROBO VIOLENTO DE VEHÍCULO', 'ROBO DE VEHÍCULO', 'ROBO DE OBJETO DE O DESDE VEHÍCULO', 'ROBO EN LUGAR HABITADO', 'ROBO EN LUGAR NO HABITADO', 'OTROS ROBOS CON FUERZA')
- "¿Cuántos robos con violencia?" → WHERE nombre_delito IN ('ROBO CON VIOLENCIA', 'ROBO CON INTIMIDACIÓN', 'ROBO POR SORPRESA', 'ROBO VIOLENTO DE VEHÍCULO')
- "¿Cuántos robos de vehículos?" → WHERE nombre_delito = 'ROBO DE VEHÍCULO'

Para preguntas donde necesites contexto sobre la fecha, hoy es día {datetime.now().day} de mes {datetime.now().month} de año {datetime.now().year}.
Nunca consultes todas las columnas. Sólo pide las columnas relevantes para la pregunta.

Responde SÓLO con la consulta SQL, sin ninguna otra explicación o texto adicional.
"""
query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", query_gen_system_prompt_text),
    MessagesPlaceholder(variable_name="messages")
])

llm_db_query_generator = ChatOpenAI(model=LLM_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0)
query_gen_runnable = query_gen_prompt | llm_db_query_generator

def query_generation_node(state: State):
    print("---GENERATING SQL QUERY---")

    response_ai_message = query_gen_runnable.invoke(state) 
    return {"messages": [response_ai_message]}

def query_check_node(state: State) -> dict[str, list[AIMessage]]:
    print("---CHECKING SQL QUERY---")
    last_message = state["messages"][-1]
    query_text_to_check = ""

    if isinstance(last_message, AIMessage) and last_message.content:
        query_text_to_check = last_message.content
    else:
        error_msg = "Error: No se encontró consulta SQL generada para verificar."
        print(error_msg)
        return {"messages": [AIMessage(content=error_msg)]} # Podría causar un bucle si no se maneja bien en el router

    messages_for_checking_invocation = [
        ("human", query_text_to_check) 
    ]
    
    print(f"--- Invoking query_check_chain with query: {query_text_to_check} ---")
    checked_query_ai_message = query_check_chain.invoke(
        {"query_messages_for_check": messages_for_checking_invocation}
    )
    
    return {"messages": [checked_query_ai_message]}


response_gen_system_prompt_text = """Eres un asistente útil que responde preguntas basándose en los resultados de una consulta SQL.
La pregunta original del usuario y los resultados de la consulta SQL se proporcionarán.
Formula una respuesta concisa y clara en español para el usuario.
Si los resultados de la consulta están vacíos o no son informativos, indícalo amablemente.

Se amable, calido y profesional con tus respuestas.
LA RESPUESTA SIEMPRE DEBE ESTAR EN FORMATO HTML.

EJEMPLO:
Claro, aquí está la información que has solicitado:
<table>
    <tr>
        <th>Nombre</th>
        <th>Edad</th>
    </tr>
</table>

"""
response_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", response_gen_system_prompt_text),
    MessagesPlaceholder(variable_name="messages")
])
llm_response_generator = ChatOpenAI(model=LLM_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0.1)
response_gen_runnable = response_gen_prompt | llm_response_generator

def final_response_generation_node(state: State):
    print("---GENERATING FINAL RESPONSE---")
    final_ai_response = response_gen_runnable.invoke(state)
    return {"messages": [final_ai_response]}

# Nodos de herramientas 
list_tables_tool_node = create_tool_node_with_fallback([list_tables_tool])
get_schema_tool_node = create_tool_node_with_fallback([get_schema_tool])
execute_query_tool_node = create_tool_node_with_fallback([db_query_tool]) # Este usa la herramienta @tool db_query_tool

# Añadir nodos al workflow
workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("list_tables_tool_node", list_tables_tool_node)

# Nodo para que el LLM elija el esquema (como antes, pero usando llm_db)
model_get_schema = llm_db.bind_tools([get_schema_tool])
workflow.add_node(
    "model_get_schema_node",
    lambda state: {"messages": [model_get_schema.invoke(state["messages"])]}
)
workflow.add_node("get_schema_tool_node", get_schema_tool_node)
workflow.add_node("query_generation_node", query_generation_node)
workflow.add_node("query_check_node", query_check_node)
workflow.add_node("execute_query_tool_node", execute_query_tool_node)
workflow.add_node("final_response_generation_node", final_response_generation_node)

# --- Lógica Condicional y Aristas --- 
def route_after_query_generation(state: State) -> Literal["query_check_node", "final_response_generation_node"]:
    print("---ROUTING AFTER QUERY GENERATION---")
    return "query_check_node"

def route_after_execution(state: State) -> Literal["final_response_generation_node", "query_generation_node"]:
    print("---ROUTING AFTER QUERY EXECUTION---")
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        if last_message.content and "Error:" in last_message.content:
            print("Error en la ejecución SQL, volviendo a generar consulta.")
            
            return "query_generation_node"
        else:
            print("Consulta SQL ejecutada, procediendo a generar respuesta final.")
            return "final_response_generation_node"
    
    return "query_generation_node"

workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool_node")
workflow.add_edge("list_tables_tool_node", "model_get_schema_node")
workflow.add_edge("model_get_schema_node", "get_schema_tool_node")
workflow.add_edge("get_schema_tool_node", "query_generation_node")

workflow.add_conditional_edges(
    "query_generation_node",
    route_after_query_generation,
    {"query_check_node": "query_check_node"}
)

workflow.add_edge("query_check_node", "execute_query_tool_node")

workflow.add_conditional_edges(
    "execute_query_tool_node",
    route_after_execution,
    {
        "final_response_generation_node": "final_response_generation_node",
        "query_generation_node": "query_generation_node"
    }
)

workflow.add_edge("final_response_generation_node", END)

# Compilar y exponer la app
db_agent_app = workflow.compile()