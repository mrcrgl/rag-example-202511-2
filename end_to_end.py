import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from os import environ
from embedding import create_embedding

SYSTEM_PROMPT_BASE = """Rolle:
Du bist ein präziser Close-Reading-Assistent für einen deutschsprachigen literarischen Text (bereitgestellt im Kontext unten).
Antworte ausschließlich auf Basis der bereitgestellten Textauszüge (“Kontextblöcke”) und zitiere kurze Belegstellen mit Chunk-IDs.

Richtlinien:
- Antworte in der Sprache der Nutzeranfrage (Standard: Deutsch).
- Nutze NUR die bereitgestellten Kontextblöcke. Ignoriere Weltwissen, wenn es dem Kontext widerspricht.
- Wenn die Information im Kontext nicht vorhanden ist, sage knapp: “Nicht im bereitgestellten Kontext.”
- Beantworte Faktenfragen direkt, gefolgt von 1–2 kurzen Belegzitaten in Anführungszeichen.
- Gib zu jedem Beleg die Chunk-ID in eckigen Klammern an, z. B. [cid: 1234].
- Fasse dich prägnant (typisch 2–6 Sätze), außer die Nutzerin/der Nutzer bittet ausdrücklich um mehr.
- Bewahre historische/archaische Orthographie der Zitate; normalisiere nur in deiner Paraphrase.
- Erfinde keine Zitate. Keine spekulativen Interpretationen ohne klaren Textbezug.

Ausgabeformat:
1) Antwort: Deine kurze, direkte Antwort.
2) Belege:
   - [cid: <ID>] “<wörtliches Kurz-Zitat>”
   - [cid: <ID>] “<wörtliches Kurz-Zitat>”
3) (Optional) Hinweise: Unsicherheiten, offene Punkte oder welche Infos im Kontext fehlten.

Kontextblöcke:
{context_blocks}

Aufgabe:
Beantworte die Nutzerfrage ausschließlich mit Bezug auf die Kontextblöcke.
"""
CHAT_MODEL = environ.get("CHAT_MODEL", "gpt-4o-mini")
MAX_CHUNK_CHARS=1200

client = openai.Client(api_key=environ.get('OPENAI_API_KEY'))
qclient = QdrantClient(host="localhost", port=6333)

def trim_text(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + " …"

def ask_chat(system_prompt: str, user_question: str) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Nutzerfrage:\n{user_question}\n\nBitte beachte die Richtlinien im System-Prompt und gib Belege mit [cid: ...] an."},
        ],
    )
    return (resp.choices[0].message.content or "") if resp.choices and resp.choices[0].message else ""

def build_context_blocks(user_prompt_vec):
    chunks = find_chunks(user_prompt_vec)
    context_blocks=[]
    for chunk in chunks.points:
        pid = getattr(chunk, "id", None)
        score = getattr(chunk, "score", None)
        payload = getattr(chunk, "payload", {}) or {}
        story = payload.get('story', '')
        text = payload.get('text', '')
        if not text:
            # Skip points without usable text
            continue
        text = trim_text(text, MAX_CHUNK_CHARS)
        context_blocks.append(
            f"### Chunk cid:{pid} (score:{score})\nStory: {story}\n{text}\n— Ende Chunk —"
        )

    return context_blocks

def build_system_prompt(context_blocks):
    return SYSTEM_PROMPT_BASE.format(context_blocks=context_blocks)

def main():
    user_prompt = "Was lernten die Brüder?"
    user_prompt_vec = create_embedding(user_prompt)
    context_blocks = build_context_blocks(user_prompt_vec)
    system_prompt = build_system_prompt("\n".join(context_blocks))

    response = ask_chat(system_prompt, user_prompt)

    print(response)

def find_chunks(user_prompt_vec):
    chunks = qclient.query_points(
        collection_name="my_collection",
        query=user_prompt_vec,
        limit=5,
        score_threshold=0.5
    )
    return chunks

if __name__ == "__main__":
    main()
