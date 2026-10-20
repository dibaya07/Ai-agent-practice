from openai import OpenAI
from dotenv import load_dotenv
import os

user_input=input(">")


load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-3.1-flash-lite",
    messages=[
        {   "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"{user_input}"
        }
    ]
)

print("Response :-> ",response.choices[0].message.content)