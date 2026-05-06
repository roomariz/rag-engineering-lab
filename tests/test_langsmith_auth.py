import config
from langsmith import Client

client = Client()

print("Connected successfully")

projects = list(client.list_projects())

print("Projects:")
for p in projects:
    print("-", p.name)
