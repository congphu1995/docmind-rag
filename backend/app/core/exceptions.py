class DocMindError(Exception):
    """Base exception for all DocMind errors."""

    pass


class ParserError(DocMindError):
    pass


class EncryptedDocumentError(ParserError):
    """Raised when a PDF is password-protected."""

    pass


class UnsupportedFileTypeError(ParserError):
    pass


class ChunkingError(DocMindError):
    pass


class EmbeddingError(DocMindError):
    pass


class VectorStoreError(DocMindError):
    pass


class IngestionError(DocMindError):
    pass


class RetrievalError(DocMindError):
    pass
