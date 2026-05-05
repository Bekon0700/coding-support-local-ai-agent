from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")
response = llm.invoke("What is a vector database? Answer in 2 sentences. give me some coding example in details.")
print(response)
