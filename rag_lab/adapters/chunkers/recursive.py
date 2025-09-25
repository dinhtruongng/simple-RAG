from typing import Iterator, List, Sequence, Union

from rag_lab.contracts.document import Chunk, Document


class RecursiveChunker:
    name = "recursive"

    def __init__(self, chunk_chars: int, overlap_chars: int):
        self.c = chunk_chars
        self.o = overlap_chars

    def split(self, docs: Union[Sequence[Document], Iterator[Document]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for d in docs:
            text = d.text
            start = 0
            while start < len(text):
                end = min(len(text), start + self.c)
                piece = text[start:end]
                chunks.append(Chunk(id=f"{d.id}:{start}", document_id=d.id, text=piece))
                start = max(end - self.o, end)
        return chunks

    def compress(
        self, chunks: Union[Sequence[Chunk], Iterator[Chunk]], max_tokens: int
    ) -> List[Chunk]:
        return list(chunks)


# print(isinstance(RecursiveChunker, IChunker))
