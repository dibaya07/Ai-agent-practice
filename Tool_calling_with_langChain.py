from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
import requests
from langchain.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage

load_dotenv()

#making the tool function
@tool
def get_weather(location:str)-> str:
    """Get current weather information for a location."""#this tells the model what the function does
    response = requests.get(f"https://wttr.in/{location}?format=3", timeout=10)
    response.raise_for_status()
    return response.text

model = init_chat_model(
    model="gemini-3.8-flash",
    model_provider="google_genai",
    api_key=os.getenv("GEMINI_API_KEY"),
)

#binding the tools with model
model_with_tools = model.bind_tools([get_weather])
user_input = input("> ") #taking user input

#Creates structured messages: system instructions and user input.
messages = [
    SystemMessage(content="""
        You are a helpful assistant.

        Rules:
        1. Answer directly if you can answer using your knowledge.
        2. If the user asks about current weather, use the get_weather tool.
        3. If no available tool can help, answer using your knowledge
           or explain your limitations.
        4. Keep answers simple and short.

        Examples:

        User: What is Python?
        Assistant: Python is a programming language.

        User: What is the current weather in London?
        Assistant: Use the get_weather tool with location="London".

        User: What is the weather in Paris?
        Assistant: Use the get_weather tool with location="Paris".
    """),
    HumanMessage(content = user_input)
    ]

#making the list of available tools to tell the model 
available_tools = {
    "get_weather" : get_weather
}

response = model_with_tools.invoke(messages) #calling the model , here model decide he need a tool or not 
messages.append(response) #storing the assistant's reply for history

if response.tool_calls: #checking is model need a tool call, if yes
    for tool_call in response.tool_calls: #here will check which tool model wanted to call
        selected_tool = available_tools[tool_call["name"]] #selecting the tool from available tool list 
        result = selected_tool.invoke(tool_call["args"]) #calling the tool for required information
        tool_message = ToolMessage(
            content=str(result), tool_call_id=tool_call["id"] #converting the tool's output into proper format
        )
        messages.append(tool_message) #storing the tools output as well
    final_output = model_with_tools.invoke(messages) #sending the tools output to model to generate the proper final answer 
    print('Assitant', final_output.content[0]['text'])
else: #if model don't need a tool call then reply what ever answer model generated.
    print("Assistant", response.content)