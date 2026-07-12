from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = Path(__file__).parent / "data" / "chroma"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="rl_papers",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
)

query = "reward shaping in reinforcement learning"
results = vectorstore.similarity_search(query, k=3)

for r in results:
    print(r.metadata["title"])
    print(r.page_content[:150])
    print("---")