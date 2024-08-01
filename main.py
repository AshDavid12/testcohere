from langchain_cohere import ChatCohere
from langchain_cohere import CohereRagRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv('.env')

cohere_api_key = os.getenv('COHERE_API_KEY')
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY not found in environment variables")

# Load the cohere chat model
try:
    cohere_chat_model = ChatCohere(cohere_api_key=cohere_api_key)
except Exception as e:
    print("Error initializing Cohere chat model:", e)
    raise

# Create the cohere rag retriever using the chat model with the web search connector
try:
    rag = CohereRagRetriever(llm=cohere_chat_model, connectors=[{"id": "web-search"}])
except Exception as e:
    print("Error initializing Cohere RAG retriever:", e)
    raise

# Debugging: Check the models available
try:
    print("Listing available models...")
    models = cohere_chat_model.client.models.list()
    print("Available models:", models)
except Exception as e:
    print("Error listing models:", e)
    raise

# Invoke the RAG retriever
try:
    docs = rag.invoke("Who founded Cohere?")
    # Print the documents
    for doc in docs[:-1]:
        print(doc.metadata)
        print("\n\n" + doc.page_content)
        print("\n\n" + "-" * 30 + "\n\n")
    # Print the final generation
    answer = docs[-1].page_content
    print(answer)
    # Print the final citations
    citations = docs[-1].metadata['citations']
    print(citations)
except Exception as e:
    print("Error invoking RAG retriever:", e)
    raise
