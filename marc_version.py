import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from os import environ

from qdrant_client.conversions.common_types import ScoredPoint
from embedding import create_embedding

load_dotenv()

client = openai.Client(api_key=environ.get('OPENAI_API_KEY'))
qclient = QdrantClient(host="localhost", port=6333)

def main():
    prompt = "Was lag unter dem Stein?"
    prompt_embedding = create_embedding(prompt)

    result = qclient.query_points(
        collection_name="my_collection",
        query=prompt_embedding,
        limit=10,
        score_threshold=0.3,
    )

    top_by_story = dict()
    for point in result.points:
        if point.payload['story'] not in top_by_story:
            top_by_story[point.payload['story']] = point
        elif top_by_story[point.payload['story']].score < point.score:
            top_by_story[point.payload['story']] = point


    for story in top_by_story:
        point = top_by_story[story]
        print(f"Score {point.score}: {point.payload['story']} \n{point.payload['text']}\n---")


if __name__ == "__main__":
    main()
