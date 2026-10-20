from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
user_input=input(">")

loader = PyMuPDFLoader('Javascript.pdf')
docs = loader.load()

#spliting text here.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents=docs)

text_content = [text.page_content for text in texts]

# Integrating Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",api_key=os.getenv("GEMINI_API_KEY"))


url = os.getenv("QDRANT_ENDPOINT")
api_key = os.getenv("QDRANT_ENDPOINT_API_KEY")


#Use this to add initial or PDF content in the database.

# qdrant = QdrantVectorStore.from_documents(
#     texts,
#     embeddings,
#     url=url,
#     api_key=api_key,
#     collection_name="test_langchain",
# )

# print(qdrant)


# Creating Retriver function. This function help us to retrive the relevant information from the database according to user prompt.
retriver = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="test_langchain",
    api_key=api_key,
    url=os.getenv("QDRANT_ENDPOINT"),
)

#Using retriver function to find similar data from database.
relevant_chunks = retriver.similarity_search(query=user_input)

context=""

#Extracting the main content from the retrived information and storing in a string .
for chunk in relevant_chunks:
    context += chunk.page_content


model = init_chat_model(
    model="gemini-3.1-flash-lite",
    model_provider="openai",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)

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

Context: {context}
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