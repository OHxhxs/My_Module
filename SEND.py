import requests

requests.get("http://192.168.0.42:5000/reserve", json={'name': 'Test User'})