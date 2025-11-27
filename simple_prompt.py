#!/usr/bin/env python3
"""
Query Qdrant using OpenAI embeddings against an existing collection.

This script:
- Embeds your query text with OpenAI "text-embedding-3-small"
- Queries a running Qdrant instance (e.g., via docker-compose)
- Returns the top-k most similar chunks stored in the collection

Prerequisites:
- Qdrant is running and the collection is already indexed (see indexing.py).
- .env contains OPENAI_API_KEY (embedding module loads it).

Examples:
- One-off query:
  python simple_prompt.py --query "Worum geht es in der Geschichte mit Betty?"

- Interactive mode:
  python simple_prompt.py --interactive

- Custom host/port/collection:
  python simple_prompt.py --host localhost --port 6333 --collection my_collection --query "Betty"
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Any, Iterable

from qdrant_client import QdrantClient
from embedding import create_embedding  # Reuse existing embedding utility (loads .env, sets OpenAI client)


def connect_qdrant(host: str, port: int) -> QdrantClient:
    return QdrantClient(host=host, port=port)


def get_points_from_query_response(resp: Any) -> Iterable[Any]:
    """
    Qdrant client may return:
      - a list[ScoredPoint] (older .search behavior), or
      - an object with `.points` (query_points response).
    Normalize to an iterable of points.
    """
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    points = getattr(resp, "points", None)
    if points is not None:
        return points
    # Fallback: try to iterate resp directly if it's iterable
    try:
        return list(resp)
    except TypeError:
        return []


def run_query(
    client: QdrantClient,
    collection: str,
    query: str,
    top_k: int,
    score_threshold: float | None = None,
):
    # Embed the query
    qvec = create_embedding(query)

    # Execute ANN search
    resp = client.query_points(
        collection_name=collection,
        query_vector=qvec,
        limit=top_k,
        filter=None,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False,
    )

    points = list(get_points_from_query_response(resp))
    return points


def format_result(point: Any, width: int = 100, max_chars: int | None = None) -> str:
    score = getattr(point, "score", None)
    payload = getattr(point, "payload", {}) or {}
    pid = getattr(point, "id", None)

    story = payload.get("story", "")
    text = payload.get("text", "")

    if max_chars is not None and isinstance(max_chars, int) and max_chars > 0:
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."

    wrapped_text = "\n\n".join(
        textwrap.fill(p, width=width, replace_whitespace=False)
        for p in text.split("\n\n")
        if p.strip()
    )

    header_bits = []
    if score is not None:
        header_bits.append(f"score={score:.4f}")
    if pid is not None:
        header_bits.append(f"id={pid}")
    if story:
        header_bits.append(f"story={story}")

    header = " | ".join(header_bits) if header_bits else "Result"
    return f"{header}\n{wrapped_text}"


def main():
    parser = argparse.ArgumentParser(description="Query Qdrant using OpenAI embeddings.")
    parser.add_argument("--host", default="localhost", help="Qdrant host (default: localhost)")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port (default: 6333)")
    parser.add_argument("--collection", default="my_collection", help="Qdrant collection name (default: my_collection)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest chunks to return")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional minimum score threshold")
    parser.add_argument("--width", type=int, default=100, help="Output wrap width")
    parser.add_argument("--max-chars", type=int, default=None, help="Truncate text to this many characters (optional)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Query string (omit to use --interactive)")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt mode")
    args = parser.parse_args()

    qclient = connect_qdrant(args.host, args.port)

    # Validate collection exists
    if not qclient.collection_exists(args.collection):
        print(
            f"Collection '{args.collection}' does not exist on {args.host}:{args.port}. "
            f"Please index data first.",
            file=sys.stderr,
        )
        sys.exit(1)

    def do_query(q: str):
        points = run_query(
            client=qclient,
            collection=args.collection,
            query=q,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
        )
        if not points:
            print("No results.")
            return

        print("\nTop results:")
        print("=" * 80)
        for i, p in enumerate(points, start=1):
            print(f"\n[{i}] {format_result(p, width=args.width, max_chars=args.max_chars)}")
            print("-" * 80)

    if args.query:
        do_query(args.query)
    elif args.interactive:
        print("Interactive mode. Type your query and press Enter. Ctrl+C to exit.")
        try:
            while True:
                q = input("\nQuery> ").strip()
                if not q:
                    continue
                do_query(q)
        except KeyboardInterrupt:
            print("\nGoodbye!")
    else:
        print("Provide --query 'your question' or use --interactive.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
