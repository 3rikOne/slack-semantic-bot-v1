import json
from openai import OpenAI

client = OpenAI()

with open("faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

out = []
for item in faq:
    q = item["question"]
    a = item["answer"]

    # IMPORTANT: embed richer text
    text_for_embedding = f"OTAZKA: {q}\nODPOVED: {a}"

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_for_embedding
    ).data[0].embedding

    out.append({
        "question": q,
        "answer": a,
        "embedding": emb
    })

with open("faq_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print(f"Saved {len(out)} embeddings to faq_embeddings.json")
