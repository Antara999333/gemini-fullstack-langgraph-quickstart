import os
import requests
from dotenv import load_dotenv
from agent.tools_and_schemas import SearchQueryList, Reflection
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.state import OverallState, QueryGenerationState, ReflectionState, WebSearchState
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.utils import (
    insert_citation_markers,
)

load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set")

if os.getenv("SEARCH_API_KEY") is None:
    raise ValueError("SEARCH_API_KEY is not set")


def search_web(query: str, num_results: int = 10):
    """Perform web search using SearchAPI.io."""
    try:
        search_url = "https://www.searchapi.io/api/v1/search"
        
        params = {
            "api_key": os.getenv("SEARCH_API_KEY"),
            "engine": "google",
            "q": query,
            "num": min(num_results, 10),  # SearchAPI.io typically limits to 10
        }
        
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract search results from SearchAPI.io response
        results = []
        if "organic_results" in data:
            for result in data["organic_results"][:num_results]:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                })
        
        return results
        
    except Exception as e:
        print(f"Search API error: {e}")
        return []


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """Generates search queries based on the user's question using OpenAI."""
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state["messages"][-1].content,
        number_queries=state["initial_search_query_count"],
    )

    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """Spawns web research nodes for each search query."""
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Performs web research using search API and OpenAI for analysis."""
    configurable = Configuration.from_runnable_config(config)
    
    # Perform actual web search
    search_results = search_web(state["search_query"], num_results=8)
    
    if not search_results:
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [f"No search results found for: {state['search_query']}"],
        }
    
    # Create sources list with URLs
    sources = []
    search_content = ""
    
    for i, result in enumerate(search_results, 1):
        source_info = {
            "url": result["url"],
            "title": result["title"],
            "snippet": result["snippet"]
        }
        sources.append(source_info)
        
        search_content += f"\n[{i}] {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n---\n"
    
    # Use OpenAI to analyze and summarize the search results
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    
    analysis_prompt = f"""
    {formatted_prompt}
    
    Based on the following search results for "{state['search_query']}", provide a comprehensive analysis:
    
    {search_content}
    
    Please synthesize the key information, highlight important findings, and note any contradictions or gaps.
    Include specific references to sources using [1], [2], etc. format.
    """
    
    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=0.1,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    response_text = response.content
    
    # Insert citation markers
    modified_text = insert_citation_markers(response_text, sources)
    
    return {
        "sources_gathered": sources,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """Analyzes web research summaries to identify knowledge gaps using OpenAI."""
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=state["messages"][-1].content,
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )

    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(state: ReflectionState, config: RunnableConfig):
    """Decides whether to continue research or finalize the answer."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = state.get("max_research_loops") or configurable.max_research_loops

    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        # Return Send objects for additional research
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """Finalizes the research summary using OpenAI."""
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=state["messages"][-1].content,
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    result = llm.invoke([HumanMessage(content=formatted_prompt)])

    # Collect all sources from the research
    all_sources = []
    for sources in state.get("sources_gathered", []):
        if isinstance(sources, list):
            all_sources.extend(sources)
        else:
            all_sources.append(sources)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": all_sources,
    }


# Build Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges("generate_query", continue_to_web_research, ["web_research"])
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges("reflection", evaluate_research, ["web_research", "finalize_answer"])
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="openai-research-agent")