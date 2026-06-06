from .schemas import DocumentSummary, PaperMetadata, SourceChunk


def paper_metadata(metadata) -> PaperMetadata:
    return PaperMetadata(
        title=metadata.title,
        authors=metadata.authors,
        year=metadata.year,
        venue=metadata.venue,
        doi=metadata.doi,
        abstract=metadata.abstract,
        publisher=metadata.publisher,
        external_url=metadata.external_url,
        reference_count=metadata.reference_count,
        citation_count=metadata.citation_count,
        fields_of_study=metadata.fields_of_study,
        metadata_confidence=metadata.metadata_confidence,
        metadata_match_score=metadata.metadata_match_score,
        metadata_source=metadata.metadata_source,
        is_enriched=metadata.is_enriched,
        keywords=metadata.keywords,
        duplicate_of=metadata.duplicate_of,
        duplicate_reason=metadata.duplicate_reason,
    )


def document_summary(document) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.document_id,
        filename=document.filename,
        pages=document.pages,
        chunks=document.chunks,
        metadata=paper_metadata(document.metadata),
    )


def source_chunk(result) -> SourceChunk:
    return SourceChunk(
        document_id=result.chunk.document_id,
        filename=result.chunk.filename,
        page=result.chunk.page,
        chunk_id=result.chunk.chunk_id,
        score=result.score,
        text=result.chunk.text,
        section=result.chunk.section,
    )


def source_chunks(results) -> list[SourceChunk]:
    return [source_chunk(result) for result in results]
