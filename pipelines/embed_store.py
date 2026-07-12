from pathlib import Path
from typing import List

import duckdb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from prefect import task, flow, get_run_logger

DB_PATH = Path(__file__).parent.parent / "data" / "papers.duckdb"
CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma"


@task
def load_papers_from_duckdb() -> List[dict]:
    """Pull all papers from DuckDB that haven't been embedded yet."""
    logger = get_run_logger()
    con = duckdb.connect(str(DB_PATH))

    # Track embedded papers separately so we don't re-embed on every run
    con.execute("""
        CREATE TABLE IF NOT EXISTS embedded_papers (
            arxiv_id VARCHAR PRIMARY KEY
        )
    """)

    rows = con.execute("""
        SELECT p.arxiv_id, p.title, p.abstract, p.url
        FROM papers p
        LEFT JOIN embedded_papers e ON p.arxiv_id = e.arxiv_id
        WHERE e.arxiv_id IS NULL
    """).fetchall()

    con.close()
    logger.info(f"Found {len(rows)} papers not yet embedded")

    return [
        {"arxiv_id": r[0], "title": r[1], "abstract": r[2], "url": r[3]}
        for r in rows
    ]


@task
def chunk_and_embed(papers: List[dict]) -> int:
    """Chunk abstracts, embed them, and store in Chroma."""
    logger = get_run_logger()

    if not papers:
        logger.info("No new papers to embed.")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []

    for paper in papers:
        chunks = splitter.split_text(paper["abstract"])
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "arxiv_id": paper["arxiv_id"],
                        "title": paper["title"],
                        "url": paper["url"],
                        "chunk_index": i,
                    },
                )
            )

    logger.info(f"Created {len(documents)} chunks from {len(papers)} papers")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        collection_name="rl_papers",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    vectorstore.add_documents(documents)

    logger.info(f"Embedded and stored {len(documents)} chunks in Chroma")

    # Mark these papers as embedded
    con = duckdb.connect(str(DB_PATH))
    for paper in papers:
        con.execute(
            "INSERT INTO embedded_papers (arxiv_id) VALUES (?) ON CONFLICT DO NOTHING",
            [paper["arxiv_id"]],
        )
    con.close()

    return len(papers)


@flow(name="embed-papers")
def embed_papers_flow():
    logger = get_run_logger()
    logger.info("Starting embedding flow")

    papers = load_papers_from_duckdb()
    count = chunk_and_embed(papers)

    logger.info(f"Flow complete. {count} papers embedded.")
    return count


if __name__ == "__main__":
    embed_papers_flow()