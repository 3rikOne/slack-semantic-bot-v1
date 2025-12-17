import os
from fastapi import Request
from fastapi import FastAPI
from pydantic import BaseModel
import json
import time
import urllib.request
import urllib.error

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel

# -------------------------
# Setup
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# -------------------------
# FAQ embeddings load
# -------------------------
with open("faq_embeddings.json", "r") as f:
    faq_data = json.load(f)

faq_embeddings = [np.array(item["embedding"]) for item in faq_data]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_reply(user_text: str) -> str:
    emb_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_text
    )
    user_emb = np.array(emb_response.data[0].embedding)

    best_sim = -1.0
    best_item = None

    for item, emb in zip(faq_data, faq_embeddings):
        sim = cosine_similarity(user_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_item = item

    THRESHOLD = 0.60
    if best_sim >= THRESHOLD and best_item:
        return best_item["answer"]
      
    system_prompt = (
        "Ak otázka NIE JE pracovná, odpovedz normálne po slovensky.\n"
        "Ak JE pracovná a nie je vo FAQ, odpovedz presne:\n"
        "'Nemám k tejto otázke odpoveď v interných FAQ.'"
    )

    chat = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    )

    return chat.choices[0].message.content.strip()
    
def is_work_question(user_text: str) -> bool:
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content":
             "Classify the user message.\n"
             "Return ONLY one word: WORK or NONWORK.\n"
             "WORK = cars/import/export/logistics/invoices/contracts/internal processes.\n"
             "NONWORK = everything else.\n"},
            {"role": "user", "content": user_text}
        ],
    )
    out = r.choices[0].message.content.strip().upper()
    return out == "WORK"

def answer_question(user_text: str) -> str:
    # 1) embed user question
    emb_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_text
    )
    user_emb = np.array(emb_response.data[0].embedding)

    # 2) find best FAQ match
    best_sim = -1.0
    best_item = None

    for item, emb in zip(faq_data, faq_embeddings):
        sim = cosine_similarity(user_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_item = item

    THRESHOLD = 0.60

    # DEBUG — MUST EXIST TEMPORARILY
    print("DEBUG best_sim:", best_sim)
    print("DEBUG best_q:", best_item["question"] if best_item else None)

    # CASE 1 — FAQ known
    if best_sim >= THRESHOLD and best_item:
        return best_item["answer"]

    # CASE 2 — WORK but not in FAQ
    if is_work_question(user_text):
        return "Nemám k tejto otázke odpoveď v interných FAQ."

    # CASE 3 — NON-WORK → normal LLM reply
    chat = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Odpovedz po slovensky. Neopakuj otázku. Odpovedz priamo."
            },
            {
                "role": "user",
                "content": user_text
            }
        ]
    )

    return chat.choices[0].message.content.strip()

def post_to_slack(channel: str, text: str):
    if not SLACK_BOT_TOKEN:
        print("Missing SLACK_BOT_TOKEN env var")
        return

    payload = {"channel": channel, "text": text}

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
            body = resp.read().decode("utf-8")
            # Helpful if something fails silently:
            # print("Slack API response:", body)
    except urllib.error.HTTPError as e:
        print("Slack HTTPError:", e.read().decode("utf-8"))
    except Exception as e:
        print("Slack post exception:", str(e))


# -------------------------
# Dedupe: prevent repeated replies
# -------------------------
PROCESSED_EVENTS = {}  # event_id -> timestamp
DEDUP_TTL_SECONDS = 60 * 10  # keep 10 min


def seen_event(event_id: str) -> bool:
    now = time.time()

    # cleanup old
    for k, ts in list(PROCESSED_EVENTS.items()):
        if now - ts > DEDUP_TTL_SECONDS:
            PROCESSED_EVENTS.pop(k, None)

    if not event_id:
        return False

    if event_id in PROCESSED_EVENTS:
        return True

    PROCESSED_EVENTS[event_id] = now
    return False

def handle_message_event(channel: str, text: str):
    reply = answer_question(text)
    post_to_slack(channel, reply)

# -------------------------
# Slack Events endpoint
# -------------------------
@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    # Process only events for THIS Slack app
    if data.get("api_app_id") and data.get("api_app_id") != os.getenv("SLACK_APP_ID"):
        return {"ok": True}

    # Slack URL verification
    # Slack sends: {"type":"url_verification","challenge":"..."}
    if data.get("type") == "url_verification" and "challenge" in data:
        return {"challenge": data["challenge"]}

    # Event callbacks
    if data.get("type") == "event_callback":
        event_id = data.get("event_id", "")
        if seen_event(event_id):
            return {"ok": True}

        event = data.get("event", {})

        # Ignore bot messages / message subtypes (edits, joins, etc.)
        if event.get("bot_id") or event.get("subtype"):
            return {"ok": True}

        # We only care about plain messages
        if event.get("type") == "message" and event.get("text") and event.get("channel"):
            channel = event["channel"]
            user_text = event["text"]

            # Ack immediately, process async
            background_tasks.add_task(handle_message_event, channel, user_text)

    return {"ok": True}


# -------------------------
# Your existing /route API (optional)
# -------------------------
class Question(BaseModel):
    message: str
    channel_id: str
    user_id: str


@app.post("/route")
def route(question: Question):
    reply = (question.message)
    return {"reply": reply}
