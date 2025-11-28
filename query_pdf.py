import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from os import environ
from end_to_end import ask_chat
from embed_parsed_segment import load_data
from qdrant_client.conversions.common_types import ScoredPoint
from embedding import create_embedding

load_dotenv()

client = openai.Client(api_key=environ.get('OPENAI_API_KEY'))
qclient = QdrantClient(host="localhost", port=6333)

SYSTEM_PROMPT_BASE = """Rolle:
Du bist ein präziser und faktengestützter Recherche-Assistent für diverse Anfragen zu veröffentlichen Texten und Dateien (bereitgestellt im Kontext unten).
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
4) Markiere relevante Stellen im Text fett (markdown)

Kontextblöcke:
{context_blocks}

Aufgabe:
Beantworte die Nutzerfrage ausschließlich mit Bezug auf die Kontextblöcke.
"""

def build_context_blocks(points, structured_data):
    blocks = []
    for point in points:

        matched_segment = None
        for segment in structured_data['segments']:
            if segment['segment_id'] == point.payload['segment_id']:
                matched_segment = segment
                break

        blocks.append(f"""### Segment (score: {point.score} cid: {point.payload['segment_id']})
Title: {point.payload['content_title']}
Heading: {point.payload['segment_headline']}
Page: {point.payload['pdf_page']}
Chunk No: {point.payload['segment_chunk_index']}
Text:
{matched_segment['text']}
- END SEGMENT""")
    return "\n".join(blocks)

def build_system_prompt(context_blocks):
    return SYSTEM_PROMPT_BASE.format(context_blocks=context_blocks)

def main():
    prompt = "Um wie viel Prozent ist die Gewaltkriminalität gestiegen?"
    prompt_embedding = create_embedding(prompt)

    result = qclient.query_points(
        collection_name="pdf_segments",
        query=prompt_embedding,
        limit=10,
        score_threshold=0.5,
    )

    # for point in result.points:
    #     print(f"Score {point.score}: {point.payload['segment_headline']} (S. {point.payload['pdf_page']}) \n{point.payload['text']}\n---")

    structured_data = load_data()
    context_blocks = build_context_blocks(result.points, structured_data)
    print(context_blocks)
    system_prompt = build_system_prompt(context_blocks)
    result = ask_chat(system_prompt, prompt)
    print(result)


if __name__ == "__main__":
    main()
