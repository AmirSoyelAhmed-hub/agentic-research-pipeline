import duckdb

con = duckdb.connect('data/papers.duckdb')
count = con.execute('SELECT COUNT(*) FROM papers').fetchone()
print(f"Total papers: {count[0]}")

titles = con.execute('SELECT title FROM papers LIMIT 3').fetchall()
for t in titles:
    print(t[0])

con.close()