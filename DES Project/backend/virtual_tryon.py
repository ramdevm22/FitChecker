
import requests
import os

HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def generate_tryon_image(prompt):
    if HF_API_KEY is None:
        raise Exception("HF_API_KEY not set")

    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.content
    else:
        raise Exception(response.text)
