import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agent.graph import graph
from agent.configuration import Configuration

# Load environment variables from .env
load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set in .env")

if os.getenv("SEARCH_API_KEY") is None:
    raise ValueError("SEARCH_API_KEY is not set in .env")


def run_agent(query: str):
    """
    Run the compiled LangGraph agent on a single query.
    """
    # Prepare input with HumanMessage object
    input_data = {"messages": [HumanMessage(content=query)]}
    
    # Create configuration for the agent
    config = {
        "configurable": {
            "query_generator_model": "gpt-4o-mini",  # or "gpt-3.5-turbo" for cheaper option
            "reflection_model": "gpt-4o-mini",
            "answer_model": "gpt-4o",  # Use GPT-4 for final answer synthesis
            "number_of_initial_queries": 3,
            "max_research_loops": 3
        }
    }

    try:
        # Run the graph with configuration
        result = graph.invoke(input_data, config=config)

        # Extract AIMessage content
        messages = result.get("messages", [])
        sources = result.get("sources_gathered", [])
        
        response_content = ""
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                response_content = msg.content
                break

        if not response_content:
            return "No answer generated."
        
        # Add sources if available
        if sources:
            response_content += "\n\n**Sources:**\n"
            for i, source in enumerate(sources, 1):
                if isinstance(source, dict):
                    title = source.get("title", "Unknown Title")
                    url = source.get("url", "No URL")
                    response_content += f"{i}. {title}\n   {url}\n"
        
        return response_content

    except Exception as e:
        print(f"Full error details: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error running agent: {e}"


def run_interactive_mode():
    """Run the agent in interactive mode."""
    print("OpenAI Research Agent")
    print("=" * 40)
    print("Type 'quit' or 'exit' to stop")
    print()
    
    while True:
        try:
            user_query = input("Enter your query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_query:
                print("Please enter a valid query.")
                continue
            
            print("\nProcessing your query...")
            print("-" * 40)
            
            answer = run_agent(user_query)
            print("\nAnswer:")
            print("=" * 40)
            print(answer)
            print("=" * 40)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Check if a query was passed as command line argument
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}")
        print("-" * 40)
        answer = run_agent(query)
        print("\nAnswer:")
        print("=" * 40)
        print(answer)
        print("=" * 40)
    else:
        # Interactive mode
        run_interactive_mode()