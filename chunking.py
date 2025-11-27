from __future__ import annotations
import re
from typing import List, Optional


class Chunk:
    story: str
    paragraphs: List[str]

    def __init__(self, story: str, paragraphs: List[str]) -> None:
        self.story = story
        self.paragraphs = paragraphs

    def __str__(self) -> str:
        # Join paragraphs with a blank line to preserve boundaries
        return f"{self.story}\n\n" + "\n\n".join(self.paragraphs)


def get_chunks(file_path: str = "data/35794-0.txt", window_size: int = 3, stride: int = 1) -> List[Chunk]:
    """
    Create overlapping chunks (default size 3, stride 1) per story.

    Behavior:
    - Each story starts on a line matching r'^\\d+\\.' (e.g., '1.', '2.', ...).
    - Produces sliding windows of size `window_size` (default 3) with step `stride` (default 1),
      so every chunk after the first overlaps the previous.
    - Stops processing at a line that starts with 'Kleine Gedichte.'.
    - Flushes and emits at story boundaries and at EOF.
    - Ignores multiple consecutive blank lines (no empty paragraphs).
    """
    text = load_data(file_path)

    current_story: Optional[str] = None
    current_paragraph_parts: List[str] = []
    story_paragraphs: List[str] = []
    chunks: List[Chunk] = []

    def flush_paragraph() -> None:
        # Combine accumulated line parts into a single paragraph and store if non-empty
        nonlocal current_paragraph_parts, story_paragraphs
        if current_paragraph_parts:
            paragraph = " ".join(part.strip() for part in current_paragraph_parts).strip()
            if paragraph:
                story_paragraphs.append(paragraph)
            current_paragraph_parts = []

    def emit_story_chunks() -> None:
        # Emit sliding windows for the current story
        nonlocal chunks, story_paragraphs, current_story
        if current_story and len(story_paragraphs) >= window_size:
            for i in range(0, len(story_paragraphs) - window_size + 1, stride):
                window = story_paragraphs[i:i + window_size]
                chunks.append(Chunk(current_story, window))
        # Reset for next story
        story_paragraphs.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")

        # New story header like "1.", "2.", etc.
        if re.match(r'^\d+\.', line):
            # Close out previous story (if any)
            flush_paragraph()
            emit_story_chunks()
            # Start new story
            current_story = line.strip()
            continue

        # Skip everything until first story header
        if current_story is None:
            continue

        # Stop at this section marker, but don't drop the last partials
        if line.startswith('Kleine Gedichte.'):
            flush_paragraph()
            emit_story_chunks()
            break

        # Paragraph boundary (blank line)
        if line.strip() == "":
            flush_paragraph()
        else:
            # Accumulate line into current paragraph (with spaces between lines)
            current_paragraph_parts.append(line.strip())

    # EOF flush: capture any remaining paragraph and story chunks
    flush_paragraph()
    emit_story_chunks()

    print(f"Processed {len(chunks)} chunks")
    return chunks


def load_data(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()
