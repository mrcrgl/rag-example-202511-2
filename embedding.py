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

    is_question = create_embedding("question")
    is_information = create_embedding("information or statement")

    vec1 = create_embedding("Betty wurde alt")
    similarity_question = cosine_similarity(vec1, is_question)
    similarity_information = cosine_similarity(vec1, is_information)

    print(f"Similarity to question: {similarity_question:.4f}")
    print(f"Similarity to information: {similarity_information:.4f}")


if __name__ == "__main__":
    main()
