import json
from openai import OpenAI

client = OpenAI()

with open("faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

out = []
for item in faq:
    q = item["question"]
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=q
    ).data[0].embedding
    out.append({"question": q, "answer": item["answer"], "embedding": emb})

with open("faq_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print(f"Saved {len(out)} embeddings to faq_embeddings.json")
