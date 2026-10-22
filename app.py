from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
import requests
from langchain.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage

load_dotenv()

@tool
def get_weather(location:str):
    """Get current weather information for a location."""
    print("calling tool")
    response = requests.get(f"https://wttr.in/{location}?format=3")
    return response.text

model = init_chat_model(
    model="gemini-3.8-flash",
    model_provider="google_genai",
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)

model_with_tools = model.bind_tools([get_weather])

user_input = input("> ")

messages = [HumanMessage(content = user_input)]


response = model_with_tools.invoke(messages)
messages.append(response)

if response.tool_calls:
    for tool_call in response.tool_calls:
         # Mapping to your actual functions
        tool_map = {"get_weather": get_weather}
        selected_tool = tool_map[tool_call["name"]]

        # Execute the tool
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                )
            )
final_response = model_with_tools.invoke(messages)
print(final_response.content)
