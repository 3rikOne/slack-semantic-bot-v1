import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAQ
with open("faq.json", "r") as f:
    faq = json.load(f)

embeddings = []

for item in faq:
    text = item["question"]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    vector = response.data[0].embedding
    embeddings.append({
        "question": item["question"],
        "answer": item["answer"],
        "embedding": vector
    })

# Save result
with open("faq_embeddings.json", "w") as f:
    json.dump(embeddings, f)

print("Embeddings generated and saved.")

