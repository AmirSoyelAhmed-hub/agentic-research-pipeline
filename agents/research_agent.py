import os
from pathlib import Path
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    timeout=30,
)
langfuse_handler = CallbackHandler()

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma"
MAX_RETRIEVAL_ATTEMPTS = 2


class AgentState(TypedDict):
    question: str
    search_query: str
    retrieved_chunks: List[dict]
    is_relevant: bool
    attempts: int
    answer: str


embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="rl_papers",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
)
llm = ChatOllama(model="llama3.2", temperature=0.2)


def retrieve(state: AgentState) -> AgentState:
    """Search Chroma for chunks relevant to the current search query."""
    query = state.get("search_query") or state["question"]
    results = vectorstore.similarity_search(query, k=4)
    chunks = [
        {
            "text": r.page_content,
            "title": r.metadata["title"],
            "url": r.metadata["url"],
        }
        for r in results
    ]
    return {**state, "retrieved_chunks": chunks, "search_query": query}


def grade_relevance(state: AgentState) -> AgentState:
    """Ask the LLM whether the retrieved chunks actually address the question."""
    context = "\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in state["retrieved_chunks"]
    )

    prompt = f"""Question: {state['question']}

Retrieved context:
{context}

Does this context contain enough information to meaningfully answer the question?
Reply with ONLY one word: YES or NO."""

    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()
    is_relevant = verdict.startswith("YES")

    return {**state, "is_relevant": is_relevant, "attempts": state.get("attempts", 0) + 1}


def rewrite_query(state: AgentState) -> AgentState:
    """Ask the LLM to reformulate the search query for a better retrieval attempt."""
    prompt = f"""The following search query did not retrieve good results for this question:

Question: {state['question']}
Previous search query: {state['search_query']}

Rewrite the search query using different keywords or phrasing that might retrieve more
relevant results from a database of reinforcement learning paper abstracts.
Reply with ONLY the new search query, nothing else."""

    response = llm.invoke(prompt)
    new_query = response.content.strip().strip('"')

    return {**state, "search_query": new_query}


def generate(state: AgentState) -> AgentState:
    """Generate a grounded answer using retrieved context."""
    context = "\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in state["retrieved_chunks"]
    )

    prompt = f"""You are a research assistant summarizing recent reinforcement learning papers.
Answer the question using ONLY the context below.

STRICT RULES:
- Cite papers using ONLY their exact title in square brackets, e.g. [Paper Title Here]
- NEVER invent author names, years, or arXiv IDs — none are provided in the context, so do not fabricate them
- If the context doesn't contain a good answer, say so honestly
- Do not add any citation details beyond the paper title

Context:
{context}

Question: {state['question']}

Answer:"""

    response = llm.invoke(prompt)
    return {**state, "answer": response.content}


def route_after_grading(state: AgentState) -> str:
    """Decide whether to generate an answer, retry retrieval, or give up."""
    if state["is_relevant"]:
        return "generate"
    if state["attempts"] >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate"  # give up retrying, answer honestly with what we have
    return "rewrite_query"


graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_relevance", grade_relevance)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade_relevance")
graph.add_conditional_edges(
    "grade_relevance",
    route_after_grading,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("generate", END)

research_agent = graph.compile()


if __name__ == "__main__":
    result = research_agent.invoke(
        {"question": "What are recent approaches to reward shaping in RL?", "attempts": 0},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"(took {result['attempts']} retrieval attempt(s), search query used: '{result['search_query']}')\n")
    print(result["answer"])
    print("\n--- Sources ---")
    for c in result["retrieved_chunks"]:
        print(f"- {c['title']} ({c['url']})")