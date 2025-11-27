from dotenv import load_dotenv
import openai
import numpy as np
from os import environ

load_dotenv()
client = openai.Client(api_key=environ.get('OPENAI_API_KEY'))

def create_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text, encoding_format="float")
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():

    input_text="""
    Die Mutter war auch entzückt und sagte:

    »Es ist gut, mein Kind, daß du nicht alle Birkenblätter fortwarfst! Hier
    haben wir Gold genug, um uns ein kleines Gut zu kaufen!«

    Betty und ihre verwitwete Mutter kauften nun wirklich ein kleines Gut.
    Sie kauften auch viele Kühe, Pferde, Ochsen, Schafe, Ziegen, Hühner,
    Gänse und Enten, und wurden sehr reich. Betty mußte das Vieh nicht mehr
    hüten, aber sie ging oft in den Wald, denn sie hoffte immer die schöne
    Waldfrau noch einmal dort zu sehen. Diese Hoffnung aber war immer
    vergebens, und Betty wurde sehr alt, ohne ihr schönes Mädchen je
    wiedergesehen zu haben.
    """

    print(len(input_text.split(" ")) / 0.7)

    vec1 = create_embedding(input_text)
    vec2 = create_embedding("Betty wurde sehr alt")
    similarity = cosine_similarity(vec1, vec2)

    print(similarity)


if __name__ == "__main__":
    main()
