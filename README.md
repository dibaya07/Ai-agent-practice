This project contains the test cases and learning process of GenAI

1. First we connect to Gemini api in ( connecting_to_LLM.py ) file

2. Built this pdf reader without using vector embedding only providing pdf content in system prompt in this file ( PDF_reader_without_vector.py )

3. Built this pdf reader using langChain and document loader in this file (PDF_loader_using_LangChain.py)

4. Built this pdf reader where we are storing the pdf content in vector database and by embedding them in this file ( PDF_loader_using_vectorDB_embedding.py )