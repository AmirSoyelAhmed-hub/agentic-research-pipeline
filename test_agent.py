from langchain_ollama import ChatOllama

# Connect to local Ollama model
llm = ChatOllama(model="llama3.2")

# Simple test call
response = llm.invoke("In one sentence, what is an AI agent?")
print(response.content)