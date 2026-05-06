import config

from langsmith import Client
client = Client()
projects = list(client.list_projects())
print("LangSmith connected - Projects:")
for p in projects:
    print("-", p.name)

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

print("Ollama preflight:")
print("OLLAMA_BASE_URL =", config.OLLAMA_BASE_URL)
print("OLLAMA_TIMEOUT =", config.OLLAMA_TIMEOUT)

llm = ChatOllama(
    model=config.CHAT_MODEL_MISTRAL,
    base_url=config.OLLAMA_BASE_URL,
    timeout=config.OLLAMA_TIMEOUT
)

response = llm.invoke([HumanMessage(content="Hello")])

print("Success!")
print(response.content)
