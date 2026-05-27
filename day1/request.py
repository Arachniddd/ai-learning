import requests

# response = requests.get("https://httpbin.org/get")

# print(response.status_code)
# print(response.json())

payload = {
    "message" : "hello"
}

response = requests.post("https://httpbin.org/post",
                         json=payload)

print(response.status_code)
print(response.json())
