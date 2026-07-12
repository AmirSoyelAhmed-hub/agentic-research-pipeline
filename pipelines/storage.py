from pathlib import Path
from typing import List

import duckdb
from prefect import task, get_run_logger

from pipelines.models import Paper

DB_PATH = Path(__file__).parent.parent / "data" / "papers.duckdb"


def init_db():
    """Create the papers table if it doesn't exist yet."""
    con = duckdb.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id VARCHAR PRIMARY KEY,
            title VARCHAR,
            abstract VARCHAR,
            authors VARCHAR,
            published TIMESTAMP,
            url VARCHAR,
            ingested_at TIMESTAMP DEFAULT current_timestamp
        )
    """)
    con.close()


@task
def save_to_duckdb(papers: List[Paper]) -> int:
    """Insert validated papers into DuckDB, skipping duplicates by arxiv_id."""
    logger = get_run_logger()
    init_db()

    con = duckdb.connect(str(DB_PATH))
    inserted = 0

    for p in papers:
        exists = con.execute(
            "SELECT 1 FROM papers WHERE arxiv_id = ?", [p.arxiv_id]
        ).fetchone()

        if exists:
            continue

        con.execute(
            """INSERT INTO papers (arxiv_id, title, abstract, authors, published, url)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [p.arxiv_id, p.title, p.abstract, ", ".join(p.authors), p.published, p.url],
        )
        inserted += 1

    con.close()
    logger.info(f"Inserted {inserted} new papers ({len(papers) - inserted} already existed)")
    return inserted