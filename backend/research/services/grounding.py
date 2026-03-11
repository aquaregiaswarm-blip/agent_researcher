"""Shared grounding utilities for Gemini API calls with Google Search.

Provides standalone functions used by GeminiClient, InternalOpsService,
and CompetitorSearchService to avoid code duplication and cross-class
private method calls.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)


def extract_grounding_metadata(response, GroundingMetadata, WebSource):
    """Extract grounding metadata (web sources) from a Gemini response.

    Args:
        response: Gemini API response object
        GroundingMetadata: GroundingMetadata dataclass (imported from gemini.py)
        WebSource: WebSource dataclass (imported from gemini.py)
    """
    try:
        if not response.candidates:
            return None

        candidate = response.candidates[0]
        grounding_metadata = getattr(candidate, 'grounding_metadata', None)
        if not grounding_metadata:
            return None

        grounding_chunks = getattr(grounding_metadata, 'grounding_chunks', [])
        if not grounding_chunks:
            search_queries = getattr(grounding_metadata, 'web_search_queries', []) or []
            if search_queries:
                return GroundingMetadata(web_sources=[], search_queries=list(search_queries))
            return None

        web_sources = []
        for chunk in grounding_chunks:
            web = getattr(chunk, 'web', None)
            if web:
                uri = getattr(web, 'uri', None) or ""
                title = getattr(web, 'title', None) or ""
                web_sources.append(WebSource(uri=uri, title=title))

        search_queries = getattr(grounding_metadata, 'web_search_queries', []) or []

        if web_sources or search_queries:
            return GroundingMetadata(
                web_sources=web_sources,
                search_queries=list(search_queries),
            )
        return None

    except Exception as e:
        logger.warning(f"Failed to extract grounding metadata: {e}")
        return None


def conduct_grounded_query(genai_client, prompt: str, query_type: str, model: str):
    """Make a single grounded Gemini call with GoogleSearch tool.

    Args:
        genai_client: Raw google.genai.Client instance
        prompt: The prompt text to send
        query_type: Label for logging and result identification
        model: Gemini model ID to use

    Returns:
        GroundedQueryResult
    """
    from .gemini import GroundedQueryResult, GroundingMetadata, WebSource

    try:
        from google.genai import types

        response = genai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        grounding_meta = extract_grounding_metadata(response, GroundingMetadata, WebSource)

        return GroundedQueryResult(
            query_type=query_type,
            text=response.text,
            grounding_metadata=grounding_meta,
            success=True,
        )

    except Exception as e:
        logger.error(f"Grounded query '{query_type}' failed: {e}")
        return GroundedQueryResult(
            query_type=query_type,
            text="",
            success=False,
            error=str(e),
        )


def merge_grounding_metadata(results: dict):
    """Merge and deduplicate grounding metadata from multiple query results.

    Args:
        results: Dict mapping query_type → GroundedQueryResult

    Returns:
        Optional[GroundingMetadata] with deduplicated sources and queries
    """
    from .gemini import GroundingMetadata, WebSource

    all_sources = []
    all_queries = []
    seen_uris = set()
    seen_queries = set()

    for result in results.values():
        if result.grounding_metadata:
            for source in result.grounding_metadata.web_sources:
                uri = source.uri if isinstance(source, WebSource) else source.get('uri', '')
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    all_sources.append(source)

            for query in result.grounding_metadata.search_queries:
                if query and query not in seen_queries:
                    seen_queries.add(query)
                    all_queries.append(query)

    if not all_sources and not all_queries:
        return None

    return GroundingMetadata(
        web_sources=all_sources,
        search_queries=all_queries,
    )


def run_parallel_grounded_queries(genai_client, queries: dict, model: str, max_workers: int = 8) -> dict:
    """Run multiple grounded queries in parallel.

    Args:
        genai_client: Raw google.genai.Client instance
        queries: Dict mapping query_type → prompt string
        model: Gemini model ID to use
        max_workers: Thread pool size

    Returns:
        Dict mapping query_type → GroundedQueryResult
    """
    from .gemini import GroundedQueryResult

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_type = {
            executor.submit(conduct_grounded_query, genai_client, prompt, query_type, model): query_type
            for query_type, prompt in queries.items()
        }

        for future in as_completed(future_to_type):
            query_type = future_to_type[future]
            try:
                results[query_type] = future.result()
            except Exception as e:
                logger.error(f"Query '{query_type}' raised exception: {e}")
                results[query_type] = GroundedQueryResult(
                    query_type=query_type,
                    text="",
                    success=False,
                    error=str(e),
                )

    return results
