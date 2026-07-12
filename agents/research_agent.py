from pathlib import Path
from typing import TypedDict, List

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma"


class AgentState(TypedDict):
    question: str
    retrieved_chunks: List[dict]
    answer: str


embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="rl_papers",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
)
llm = ChatOllama(model="llama3.2", temperature=0.2)


def retrieve(state: AgentState) -> AgentState:
    """Search Chroma for chunks relevant to the question."""
    results = vectorstore.similarity_search(state["question"], k=4)
    chunks = [
        {
            "text": r.page_content,
            "title": r.metadata["title"],
            "url": r.metadata["url"],
        }
        for r in results
    ]
    return {**state, "retrieved_chunks": chunks}


def generate(state: AgentState) -> AgentState:
    """Generate a grounded answer using retrieved context."""
    context = "\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in state["retrieved_chunks"]
    )

    prompt = f"""You are a research assistant summarizing recent reinforcement learning papers.
Answer the question using ONLY the context below. Cite paper titles in your answer.
If the context doesn't contain a good answer, say so honestly.

Context:
{context}

Question: {state['question']}

Answer:"""

    response = llm.invoke(prompt)
    return {**state, "answer": response.content}


graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

research_agent = graph.compile()


if __name__ == "__main__":
    result = research_agent.invoke({"question": "What are recent approaches to reward shaping in RL?"})
    print(result["answer"])
    print("\n--- Sources ---")
    for c in result["retrieved_chunks"]:
        print(f"- {c['title']} ({c['url']})")