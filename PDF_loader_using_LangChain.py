from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyMuPDFLoader

load_dotenv()
user_input=input(">")

model = init_chat_model(
    model="gemini-3.1-flash-lite",
    model_provider="openai",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)

pdf_text=""
loader = PyMuPDFLoader('Javascript.pdf')
docs = loader.load()

for doc in docs:
    pdf_text += doc.page_content

system_prompt=f"""
You are a helpful assistant. Answer in under 50 words.

Use only the provided PDF content.

Rules:
- Always Answer within 50 words.
- If the user asks any question, give a short overview.
- Ask if they want to explore a topic or page in more detail.
- If the user asks for a specific topic, explain only that topic.
- Never dump the entire PDF content at once.
- If the user asks for a page, provide a summary of that page.

Examples:

User: What does this PDF contain?
Assistant: This PDF contains JavaScript learning notes covering basics, arrays, functions, DOM, events, and APIs. Would you like to explore a specific topic?

User: Tell me about arrays.
Assistant: Arrays store multiple values in a single variable. You can access values using indexes and use methods like push(), pop(), and slice().

User: Give me all the details about arrays.
Assistant: I can explain arrays step by step, including array methods and examples. Would you like to start with the basics?

PDF content:
{pdf_text}
"""
messages=[
        {   "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

response = model.invoke(messages)

print ( response.content)