from datetime import datetime, timezone
from typing import List, Optional
import time

import arxiv
from pydantic import ValidationError
from prefect import flow, task, get_run_logger

from pipelines.models import Paper
from pipelines.storage import save_to_duckdb


@task(retries=3, retry_delay_seconds=10)
def fetch_papers(query: str, max_results: int = 20) -> List[arxiv.Result]:
    """Fetch raw results from arXiv. Retries automatically on failure."""
    logger = get_run_logger()
    logger.info(f"Fetching up to {max_results} papers for query: {query}")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client()
    results = list(client.results(search))
    logger.info(f"Fetched {len(results)} raw results")
    return results


@task
def validate_papers(raw_results: List[arxiv.Result]) -> List[Paper]:
    """Validate and clean raw arXiv results into typed Paper records."""
    logger = get_run_logger()
    valid_papers = []
    skipped = 0

    for r in raw_results:
        try:
            paper = Paper(
                arxiv_id=r.entry_id.split("/")[-1],
                title=r.title,
                abstract=r.summary,
                authors=[a.name for a in r.authors],
                published=r.published,
                url=r.entry_id,
            )
            valid_papers.append(paper)
        except ValidationError as e:
            skipped += 1
            logger.warning(f"Skipped invalid record: {e}")

    logger.info(f"Validated {len(valid_papers)} papers, skipped {skipped}")
    return valid_papers


@flow(name="arxiv-rl-ingest")
def arxiv_ingest_flow(query: str = 'cat:cs.LG AND abs:"reinforcement learning"', max_results: int = 20):
    logger = get_run_logger()
    logger.info("Starting arXiv ingestion flow")

    raw_results = fetch_papers(query=query, max_results=max_results)
    papers = validate_papers(raw_results)
    inserted = save_to_duckdb(papers)

    logger.info(f"Flow complete. {inserted} new papers saved to DuckDB.")
    return papers


if __name__ == "__main__":
    result = arxiv_ingest_flow()
    for p in result[:3]:
        print(p.title)