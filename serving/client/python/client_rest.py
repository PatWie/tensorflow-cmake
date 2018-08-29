import requests

url = "http://localhost:8501/v1/models/simple_model:predict"

payload = {'instances': [1.0]}
response = requests.post(url, json=payload)

print(response.text)
print(response.status_code, response.reason)
