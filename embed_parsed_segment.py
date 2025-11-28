import json
from pprint import pprint
from embedding import create_embedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

qclient = QdrantClient(host="localhost", port=6333)

def load_data():
    with open('parsed_pks/document_structure.json') as json_data:
        data = json.load(json_data)
    return data

def chunk_text(text, max_tokens=250):
    chunks = []
    max_words = max_tokens / 0.7
    words = text.split(" ")

    chunk_words = []
    for word in words:
        chunk_words.append(word)

        if len(chunk_words) >= max_words:
            chunks.append(" ".join(chunk_words))
            chunk_words = []

    if len(chunk_words) > 0:
        chunks.append(" ".join(chunk_words))

    return chunks

def derive_points():
    data = load_data()

    points = []
    metadata = {
        "file": data.get('pdf', "N/A")
    }
    segments = data['segments']
    print(f"Number of segments: {len(segments)}")
    for segment in segments:
        segment_id = segment['segment_id']
        segment_text = segment['text']
        segment_metadata = segment['meta']
        segemtn_meta_page = segment_metadata['page']
        segemtn_meta_headline = segment_metadata['headline']

        if segment_text is None:
            print("Segment text is None")
            continue

        chunks = chunk_text(segment_text)
        for index, chunk in enumerate(chunks):
            embedding = create_embedding(chunk)

            """
            Vector Item
            Vector: Text & Table
            Metadata
            segment_id
            segment_headline
            pdf_file_ref
            pdf_page
            content_release date*
            content_type*
            content_author*
            """
            points.append(PointStruct(
                id=segment_id,
                vector=embedding,
                payload={
                    "segment_id": segment_id,
                    "segment_chunk_index": index,
                    "segment_headline": segemtn_meta_headline,
                    "pdf_file_ref": metadata["file"],
                    "pdf_page": segemtn_meta_page,
                    "content_release_date": segment_metadata['release_date'],
                    "content_type": segment_metadata['type'],
                    "content_title": segment_metadata['title'],
                    "text": segment_text,
                },
            ))

    return points

def main():
    if not qclient.collection_exists("pdf_segments"):
       qclient.create_collection(
          collection_name="pdf_segments",
          vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
       )

    points = derive_points()
    qclient.upsert(collection_name="pdf_segments", points=points)
    print("Points upserted successfully")

if __name__ == "__main__":
    main()
