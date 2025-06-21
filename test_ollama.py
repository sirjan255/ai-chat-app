from langchain_community.llms import Ollama

llm = Ollama(model="llama3") 
response = llm.invoke("Hello")
print(response)  # Should print immediately