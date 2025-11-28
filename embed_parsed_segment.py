import json
from pprint import pprint
from embedding import create_embedding



def load_data():
    with open('parsed_pks/document_structure.json') as json_data:
        data = json.load(json_data)
    return data

def main():
    data =load_data()

    points = []
    metadata = {
        "file": getattr(data, 'pdf', None)
    }
    segments = getattr(data, 'segments', [])
    for segment in segments:
        segment_id = getattr(segment, 'segment_id', None)
        segment_text = getattr(segment, 'text', None)
        segment_metadata = getattr(segment, 'metadata', {})
        segemtn_meta_page = getattr(segment_metadata, 'page', None)
        segemtn_meta_headline = getattr(segment_metadata, 'headline', None)

        if segment_text is  None:
            continue

        embedding = create_embedding(segment_text)

        points.append(PointStruct(
            id=segment_id,
            vector=embedding,
            payload={
                "text": text, "story": chunk.story},
        ))



    pprint(data)


if __name__ == "__main__":
    main()
