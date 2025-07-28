import requests

url = 'http://127.0.0.1:5000/train'
data = {
    'csv_path': 'C:/Users/megha/OneDrive/Desktop/pydroix/CEAS_08.csv'
}

response = requests.post(url, json=data)
print('Status code:', response.status_code)
print('Response:', response.json()) 