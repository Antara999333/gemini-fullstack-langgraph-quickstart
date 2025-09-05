import argparse
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import graph

# Load environment variables from .env
load_dotenv()

def main() -> None:
    """Run the LangGraph research agent from the command line."""
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="Number of initial search queries",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        help="Maximum number of research loops",
    )
    parser.add_argument(
        "--query-model",
        default=os.getenv("QUERY_GENERATOR_MODEL", "gemini-1"),
        help="Model used for generating initial search queries",
    )
    parser.add_argument(
        "--reflection-model",
        default=os.getenv("REFLECTION_MODEL", "gemini-1"),
        help="Model used for reflection/follow-up queries",
    )
    parser.add_argument(
        "--answer-model",
        default=os.getenv("ANSWER_MODEL", "gemini-1"),
        help="Model used for final answer",
    )
    args = parser.parse_args()

    # Initialize the state for the agent
    state = {
        "messages": [HumanMessage(content=args.question)],
        "initial_search_query_count": args.initial_queries,
        "max_research_loops": args.max_loops,
        "query_generator_model": args.query_model,
        "reflection_model": args.reflection_model,
        "reasoning_model": args.answer_model,
    }

    # Run the graph
    result = graph.invoke(state)
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


if __name__ == "__main__":
    main()
