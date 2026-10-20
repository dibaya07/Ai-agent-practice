from openai import OpenAI
from dotenv import load_dotenv
import os
import pymupdf

#Taking user Input
user_input=input(">")

#Connect to the LLM
load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

#Load the PDF
pdf_text = ""
doc = pymupdf.open("Javascript.pdf")
for page in doc:
    pdf_text += page.get_text()
# print(pdf_text)

#Sending prompt to LLM
response = client.chat.completions.create(
    model="gemini-3.1-flash-lite",
    messages=[
        {   "role": "system",
            "content": f"You are a helpful assistant.You always give your answers under 50 words and only from these mentioned data {pdf_text}"
        },
        {
            "role": "user",
            "content": f"{user_input}"
        }
    ]
)

print("Response :-> ",response.choices[0].message.content)