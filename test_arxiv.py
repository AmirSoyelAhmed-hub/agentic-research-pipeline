import arxiv

search = arxiv.Search(
    query='cat:cs.LG AND abs:"reinforcement learning"',
    max_results=5,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

client = arxiv.Client()
for result in client.results(search):
    print(result.title)
    print(result.published)
    print(result.entry_id)
    print("---")