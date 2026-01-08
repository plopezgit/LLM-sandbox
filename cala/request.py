import requests

url = "https://api.cala.ai/v1/knowledge/search"
query = "How to capture a Lubina in a calm Mediterranean sea? Give three popular tips, and one secret tip."

payload = { "input": query }

headers = {
    "X-API-KEY": "xxx",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())