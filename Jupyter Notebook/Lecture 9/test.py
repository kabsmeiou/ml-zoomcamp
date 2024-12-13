import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {
  'url': 'http://bit.ly/mlbookcamp-pants'
}

r = requests.post(url, json=data).json()
print(r)