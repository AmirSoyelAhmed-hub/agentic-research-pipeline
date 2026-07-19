import duckdb

con = duckdb.connect('data/papers.duckdb')

count = con.execute('SELECT COUNT(*) FROM papers').fetchone()
print(f"Total papers in DuckDB: {count[0]}")

print("\nMost recently ingested papers (by ingested_at):")
rows = con.execute("""
    SELECT title, published, ingested_at 
    FROM papers 
    ORDER BY ingested_at DESC 
    LIMIT 10
""").fetchall()
for title, published, ingested_at in rows:
    print(f"[{ingested_at}] {title} (published: {published})")

con.close()