from os import environ
from openai.types import CreateEmbeddingResponse, Embedding, EmbeddingModel
import openai
from dotenv import load_dotenv
from pprint import pprint
import numpy as np
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models.models import Indexes
from qdrant_client.models import VectorParams, Distance, PointStruct
import chunking

load_dotenv()

client = openai.Client(api_key=environ.get('OPENAI_API_KEY'))
qclient = QdrantClient(host="localhost", port=6333)

def main():
    if not qclient.collection_exists("my_collection"):
       qclient.create_collection(
          collection_name="my_collection",
          vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
       )

    start = datetime.now()

    chunks = chunking.get_chunks()

    for index, chunk in enumerate(chunks):
        length = len("\n\n".join(chunk.paragraphs).split(" "))
        print(f"Processing chunk: {index} ({length})")

        text = "\n\n".join(chunk.paragraphs)

        response = client.embeddings.create(model= "text-embedding-3-small",
        input=text,
        encoding_format= "float")

        points = [
            PointStruct(
                id=index,
                vector=response.data[0].embedding,
                payload={
                    "text": text, "story": chunk.story},
            )

        ]
        qclient.upsert(collection_name="my_collection", points=points)

        qclient.query_points(collection_name="my_collection", query_vector=response.data[0].embedding, limit=1, filter=None)

    end = datetime.now()
    print(f"Time taken: {end - start}")


if __name__ == "__main__":
    main()
