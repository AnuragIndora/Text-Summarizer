import requests

response = requests.post(
    "http://localhost:8000/summarize",
    json={"input_text": "The quick brown fox jumps over the lazy dog."}
)

print(response.status_code)  # Should be 200
print(response.json())       # Should show {"summary": "..."}
