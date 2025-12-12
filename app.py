from fastapi import FastAPI, Request
from pydantic import BaseModel
import urllib.request
import urllib.error
import json
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------
# INIT
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# --------------------------------------------------
# LOAD FAQ EMBEDDINGS
# --------------------------------------------------

with open("faq_embeddings.json", "r") as f:
    faq_data = json.load(f)

faq_embeddings = [np.array(item["embedding"]) for item in faq_data]

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# --------------------------------------------------
# CORE LOGIC (FAQ → LLM)
# --------------------------------------------------

def route_logic(user_text: str) -> str:
    # 1) Embed user question
    emb_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_text
    )
    user_emb = np.array(emb_response.data[0].embedding)

    # 2) Find best FAQ match
    best_sim = -1.0
    best_item = None

    for item, emb in zip(faq_data, faq_embeddings):
        sim = cosine_similarity(user_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_item = item

    THRESHOLD = 0.80

    # CASE 1 — FAQ HIT
    if best_sim >= THRESHOLD:
        return best_item["answer"]

    # CASE 2/3 — LLM DECISION
    system_prompt = (
        "Si firemný chatbot pre spoločnosť, ktorá rieši dovoz a vývoz áut, logistiku, "
        "zmluvy, fakturáciu a interné procesy.\n\n"

        "PRAVIDLÁ:\n"
        "- Ak je otázka pracovná (autá, export, dovoz, logistika, fakturácia, zmluvy, interné procesy) "
        "a FAQ nemá odpoveď → odpovedz presne vetou: "
        "'Nemám k tejto otázke odpoveď v interných FAQ.'\n\n"

        "- Ak otázka NIE JE pracovná → odpovedz normálne ako ChatGPT po slovensky, "
        "ale môžeš byť trochu uštipačný, priamy,politicky zaujatý v prospech pravicových názorov a si vtipný (vulgarizmy sú povolené). "
        "Najprv daj krátku vtipnú poznámku (môže ale nemusí byť politická), potom normálnu odpoveď.\n\n"

        "- Nepíš nič o tom, že existujú pravidlá, role, prompt alebo FAQ. "
        "Len odpovedz priamo.\n"
    )

    chat_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    )

    return chat_response.choices[0].message.content.strip()

# --------------------------------------------------
# SLACK EVENTS ENDPOINT
# --------------------------------------------------

@app.post("/slack/events")
async def slack_events(request: Request):
    data = await request.json()
    print("SLACK EVENT:", data)

    # URL verification
    if "challenge" in data:
        return {"challenge": data["challenge"]}

    if data.get("type") != "event_callback":
        return {"ok": True}

    event = data.get("event", {})

    # Prevent infinite loops
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return {"ok": True}

    if event.get("type") == "message" and event.get("text") and event.get("channel"):
        channel = event["channel"]
        user_text = event["text"]

        reply_text = route_logic(user_text)

        payload = {
            "channel": channel,
            "text": reply_text
        }

        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:
                resp.read()
        except urllib.error.HTTPError as e:
            print("Slack API error:", e.read())

    return {"ok": True}

# --------------------------------------------------
# OPTIONAL API ROUTE (Make / Testing)
# --------------------------------------------------

class Question(BaseModel):
    message: str

@app.post("/route")
def route(question: Question):
    reply = route_logic(question.message)
    return {"reply": reply}
