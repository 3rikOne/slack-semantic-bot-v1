from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
import urllib.request
import urllib.error
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import os

# -----------------------
# Setup
# -----------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# -----------------------
# Load FAQ embeddings at startup
# -----------------------
with open("faq_embeddings.json", "r") as f:
    faq_data = json.load(f)

faq_embeddings = [np.array(item["embedding"]) for item in faq_data]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_reply(user_text: str) -> dict:
    """
    Returns a dict with:
      - source: "faq" or "llm"
      - similarity: float
      - reply: str
      - matched_question: optional
    """
    # 1) Embedding for user question
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

    # CASE 1 — FAQ known
    if best_sim >= THRESHOLD and best_item:
        return {
            "source": "faq",
            "similarity": best_sim,
            "matched_question": best_item.get("question", ""),
            "reply": best_item.get("answer", "")
        }

    # CASE 2/3 — No FAQ match → LLM decides work vs non-work
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

    reply_text = chat_response.choices[0].message.content.strip()

    return {
        "source": "llm",
        "similarity": best_sim,
        "reply": reply_text
    }


def slack_post_message(channel: str, text: str):
    """
    Posts a message to Slack channel.
    """
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
            raw = resp.read().decode("utf-8")
            # Optional: log Slack API response if debugging
            # print("Slack response:", raw)
    except urllib.error.HTTPError as e:
        print("Slack HTTPError:", e.read().decode("utf-8"))
    except Exception as e:
        print("Slack post error:", str(e))


def process_slack_message(channel: str, user_text: str):
    """
    Runs semantic match + LLM and posts the final reply to Slack.
    """
    try:
        result = build_reply(user_text)
        reply_text = result["reply"]
        slack_post_message(channel, reply_text)
    except Exception as e:
        print("Processing error:", str(e))
        slack_post_message(channel, "Nastala chyba pri spracovaní otázky.")


# -----------------------
# Slack Events endpoint
# -----------------------
@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    print("SLACK PAYLOAD:", data)

    # Slack URL verification
    if "challenge" in data:
        return {"challenge": data["challenge"]}

    # Handle event callbacks
    if data.get("type") == "event_callback":
        event = data.get("event", {})

        # Ignore bot messages (prevents loops)
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return {"ok": True}

        # Only handle messages with text in a channel
        if event.get("type") == "message" and event.get("text") and event.get("channel"):
            channel = event["channel"]
            user_text = event["text"].strip()

            # Run processing in background (Slack requires fast response)
            background_tasks.add_task(process_slack_message, channel, user_text)

    return {"ok": True}


# -----------------------
# Optional: keep /route for manual testing
# -----------------------
class Question(BaseModel):
    message: str
    channel_id: str = ""
    user_id: str = ""


@app.post("/route")
def route(question: Question):
    return build_reply(question.message)
