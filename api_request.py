import requests
from PIL import Image
from io import BytesIO
import os

API_KEY = '29154084-59fd978d2443c1138458d6cfd'
URL = 'https://pixabay.com/api/'

params = {
    'key': API_KEY,
    'q': 'people',               # Suchbegriff
    'image_type': 'photo',      # Bildtyp: photo, illustration, vector
    'per_page': 20,             # Anzahl der Ergebnisse pro Seite
    'category': 'people',       # Kategorie: people, animals, nature, food, architecture
}

response = requests.get(URL, params=params)

#war Suche erfolgreich
if response.status_code == 200:
    data = response.json()
    i = 0
    for hit in data['hits']:
        
        tag = hit['tags'].split(',')[0]
        path = os.path.join('images', tag)
        if not os.path.exists(path):
            os.mkdir(path)
        
        img = requests.get(hit['largeImageURL'])
        img = Image.open(BytesIO(img.content))
        img.save(os.path.join(path, tag+(str(i))+".jpg"), quality=30) #images/tag/tag1.jpg
        
        print(hit['tags'].split(',')[0])
        print(f"Bild-URL: {hit['largeImageURL']}")
        print('-' * 40)
        
        i+=1
else:
    print(f"Fehler: {response.status_code}")