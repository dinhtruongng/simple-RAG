from typing import Iterator, List, Protocol, Sequence, Union, runtime_checkable

from rag_lab.contracts.document import Chunk, Document


@runtime_checkable
class IChunker(Protocol):
    """Protocol defining the interface for document chunking operations.

    This interface provides methods for splitting documents into chunks and
    optionally compressing them to fit within token limits.
    """

    name: str

    def split(self, docs: Union[Sequence[Document], Iterator[Document]]) -> List[Chunk]:
        """Split documents into chunks.

        Args:
            docs: A sequence or iterator of Document objects to be chunked.
                  Supports both Sequence[Document] and Iterator[Document] for memory efficiency.

        Returns:
            List[Chunk]: A list of Chunk objects, each containing the chunked text
                        and metadata including the original document ID.

        Note:
            Each Chunk already contains the document ID, so there's no need to
            preserve alignment between input documents and output chunks.

        Raises:
            ValueError: If any document is invalid or cannot be processed.
            RuntimeError: If chunking fails due to resource constraints.
        """
        ...

    def compress(
        self, chunks: Union[Sequence[Chunk], Iterator[Chunk]], max_token: int
    ) -> List[Chunk]:
        """Compress chunks to fit within a maximum token limit.

        This optional method allows for context compression to reduce the size
        of chunks while preserving important information.

        Args:
            chunks: A sequence or iterator of Chunk objects to compress.
            max_token: Maximum number of tokens allowed per chunk.

        Returns:
            List[Chunk]: A list of compressed Chunk objects. If compression is not
                        applicable or needed, implementations should return the
                        original chunks unchanged (pass-through behavior).

        Note:
            This is an optional hook with pass-through default behavior in implementations.
            Implementations may choose to merge, truncate, or summarize chunks.

        Raises:
            ValueError: If max_token is invalid (e.g., negative or zero).
        """
        ...
