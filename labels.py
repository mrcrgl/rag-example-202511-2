from dotenv import load_dotenv
import openai
import numpy as np
from os import environ

load_dotenv()
client = openai.Client(api_key=environ.get("OPENAI_API_KEY"))

MODEL = "text-embedding-3-small"

# Curated seed examples per class (multilingual, varied forms, punctuation-robust)
LABEL_SEEDS = {
    "question": [
        # English
        "What is this about?",
        "How do I install it?",
        "Why is the server slow?",
        "When will it be ready?",
        "Where can I find the logs?",
        "Who is responsible for this task?",
        "Which option should I choose?",
        "Is this the correct approach?",
        "Are there any known issues?",
        "Does it support Windows?",
        "Can you help me with this?",
        "Could you explain that?",
        "Should I update now?",
        "Would this work for large files",
        "Has anyone tried this?",
        "Have you seen this error?",
        "Will this break compatibility?",
        "I wonder how this works",
        "Any idea why this fails",
        # German
        "Was ist das?",
        "Wie kann ich das installieren?",
        "Warum ist der Server langsam?",
        "Wann ist es fertig?",
        "Wo finde ich die Logs?",
        "Wer ist dafür verantwortlich?",
        "Ist das der richtige Ansatz?",
        "Gibt es bekannte Probleme?",
        "Kannst du mir dabei helfen?",
        "Sollte ich jetzt aktualisieren?",
    ],
    "statement": [
        # English
        "Here is the information you requested.",
        "This is a statement, not a question.",
        "The sky is blue.",
        "It works as expected.",
        "I will go to the store later.",
        "There are three steps to follow.",
        "FYI: the server will restart tomorrow.",
        "Note: this method is deprecated.",
        "This document explains the process in detail.",
        "In summary, the results look good.",
        "Set the environment variable and restart the app.",
        # German
        "Betty wurde alt.",
        "Das ist eine Tatsache.",
        "Hier sind die Informationen.",
        "Der Server wird morgen neu gestartet.",
        "Ich habe das gestern gemacht.",
        "Es funktioniert wie erwartet.",
        "Die Anleitung beschreibt den Prozess.",
        "Zusammengefasst sieht das gut aus.",
        "Bitte beachte die folgenden Punkte.",
    ],
    # Optional: add more classes later, e.g. "command"
    # "command": [
    #     "Installiere Python 3.12.",
    #     "Run the migration now.",
    #     "Please create a new branch and open a PR.",
    #     "Setze die Variable x auf 10.",
    # ],
}

def embed_texts(texts):
    """Embeds one or many texts and returns an array of vectors (n, d)."""
    if isinstance(texts, str):
        texts = [texts]
    resp = client.embeddings.create(
        model=MODEL,
        input=texts,
        encoding_format="float",
    )
    # Ensure stable ordering by sorting by index if provided
    vectors = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
    return np.array(vectors, dtype=np.float32)

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def build_label_centroids(label_seeds: dict[str, list[str]]):
    """Build a normalized centroid vector per label from multiple seed examples."""
    labels = []
    centroids = []
    for label, seeds in label_seeds.items():
        seed_vecs = l2_normalize(embed_texts(seeds))
        centroid = seed_vecs.mean(axis=0)
        # Normalize centroid too
        norm = np.linalg.norm(centroid)
        if norm == 0:
            norm = 1.0
        centroid = centroid / norm
        labels.append(label)
        centroids.append(centroid)
    return labels, np.stack(centroids, axis=0)

def classify(text: str, labels: list[str], centroids: np.ndarray):
    """Return (label, score, all_scores) using cosine similarity to centroids."""
    v = l2_normalize(embed_texts(text))[0]
    sims = centroids @ v  # cosine similarity because everything is normalized
    idx = int(np.argmax(sims))
    return labels[idx], float(sims[idx]), sims

def main():
    # Build centroids once (you can cache them in a real app)
    labels, centroids = build_label_centroids(LABEL_SEEDS)

    # Some example inputs to test classification
    test_inputs = [
        "Betty wurde alt.",
        "Was ist der Sinn des Lebens?",
        "Kannst du mir sagen, wann der Zug ankommt?",
        "FYI: Der Server wird morgen neu gestartet.",
        "Wie kann ich Python installieren?",
        "Ich frage mich, wie das funktioniert.",
        "Setze die Variable x auf 10.",
        "Warum ist der Himmel blau",
        "Die Antwort ist 42.",
    ]

    print("Centroids for labels:", labels)
    print()

    for text in test_inputs:
        label, score, sims = classify(text, labels, centroids)
        # Show top-2 for transparency
        order = np.argsort(sims)[::-1]
        top2 = [(labels[i], float(sims[i])) for i in order[:2]]
        print(f"Text: {text}")
        print(f" Predicted: {label} (score={score:.4f})")
        print(f" Top-2: {top2}")
        print()

if __name__ == "__main__":
    main()
