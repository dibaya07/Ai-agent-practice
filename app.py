from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import operator

load_dotenv()

model = init_chat_model(
    model="gemini-3.8-flash",
    model_provider="google_genai",
    api_key=os.getenv("GEMINI_API_KEY"),
)

@tool
def multiply(a:int, b:int)-> int:
    """Multipling the arguments"""
    return a * b

@tool
def add( a:int, b:int) -> int:
    """Adding the arguments"""
    return a + b

tools=[multiply,add]

model_with_tools =  model.bind_tools(tools)

class MessageState(TypedDict):
     messages: Annotated[list[AnyMessage], operator.add]

system_prompt = """
you are an ai assistant, you see the user input if you are able to give the response you will give the response other you look for tools and if there is any relevant tools available in the available tools who is capable of giving the answer you will call that tool
available tools:
"""

user_input= input("> ")

def model_node(state:MessageState):
    """LLM decides whether to call a tool or not, and if we need to call a tool then which tool to call"""
    
    response = model_with_tools.invoke(state["messages"][SystemMessage(content=system_prompt),HumanMessage(content=user_input)])
    print("res", response)
    # return 

# def tool_node(state: MessageState):
#     """This will decide which if model need to call which tool to call."""


graph_builder = StateGraph(MessageState)

graph_builder.add_node("model_node",model_node)
# graph_builder.add_node("tool_node",tool_node)

graph_builder.add_edge(START,"model_node")



graph = graph_builder.compile()

print(graph)





















# def model_node(state: MessageState):
#     """LLM decides whether to call a tool or not, And if we need to call a tool then which tool to call"""
#     response = model_with_tools.invoke(state["messages"])
#     print("model_node_response", response)
#     return {"messages": [response]}

# graph_builder = StateGraph(MessageState)

# graph_builder.add_node("model_node", model_node)
# graph_builder.add_node("tool_node", ToolNode(tools))

# graph_builder.add_edge(START,"model_node")

# graph_builder.add_conditional_edges(
#     "model_node",
#     tools_condition,
#      {"tools": "tool_node", "__end__": END}
# )

# graph_builder.add_edge("tool_node","model_node")
# # def route_query

# graph = graph_builder.compile()

# user_query = input("> ")
# result = graph.invoke({
#     "messages": [
#         SystemMessage(
#             content="Use the add and multiply tools for arithmetic questions. "
#                     "Answer other questions directly. Keep answers short."
#         ),
#         HumanMessage(content=user_query)
#     ]
# })

# print(result["messages"][-1].content)
