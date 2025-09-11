from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def create_short_urls(sources: List[Dict[str, Any]], id: int) -> Dict[str, str]:
    """
    Create a map of original URLs to short URLs with a unique id for each URL.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    
    Args:
        sources: List of source dictionaries containing 'url' key
        id: Unique identifier for this batch of URLs
        
    Returns:
        Dict mapping original URLs to shortened URLs
    """
    prefix = f"https://search-results.com/id/"
    
    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, source in enumerate(sources):
        url = source.get('url', '')
        if url and url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"
    
    return resolved_map


def insert_citation_markers(text: str, sources: List[Dict[str, Any]]) -> str:
    """
    Inserts citation markers into text for OpenAI responses.
    
    Since OpenAI doesn't provide automatic citation indices like Gemini,
    this function adds citation markers at the end of the text or
    replaces existing [1], [2] style markers with proper links.
    
    Args:
        text: The text to add citations to
        sources: List of source dictionaries with 'title', 'url' keys
        
    Returns:
        Text with citation markers inserted
    """
    if not sources:
        return text
    
    # If text already has [1], [2] style markers, replace them with proper links
    modified_text = text
    for i, source in enumerate(sources, 1):
        title = source.get('title', f'Source {i}')
        url = source.get('url', '#')
        
        # Replace [1], [2] etc with proper markdown links
        marker_pattern = f'[{i}]'
        link_replacement = f'[{title}]({url})'
        
        if marker_pattern in modified_text:
            modified_text = modified_text.replace(marker_pattern, link_replacement)
    
    return modified_text


def format_sources_for_citation(sources: List[Dict[str, Any]]) -> str:
    """
    Format sources into a citation string.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted citation string
    """
    if not sources:
        return ""
    
    citations = []
    for i, source in enumerate(sources, 1):
        title = source.get('title', f'Source {i}')
        url = source.get('url', '#')
        citations.append(f"[{i}] [{title}]({url})")
    
    return "\n".join(citations)


def extract_numbered_citations(text: str) -> List[int]:
    """
    Extract numbered citations like [1], [2] from text.
    
    Args:
        text: Text to search for citations
        
    Returns:
        List of citation numbers found
    """
    import re
    
    # Find all [number] patterns
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, text)
    
    # Convert to integers and return unique sorted list
    return sorted(list(set(int(match) for match in matches)))


def create_citation_mapping(sources: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    """
    Create a mapping of citation numbers to source information.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Dictionary mapping citation numbers to source info
    """
    mapping = {}
    for i, source in enumerate(sources, 1):
        mapping[i] = {
            'title': source.get('title', f'Source {i}'),
            'url': source.get('url', '#'),
            'snippet': source.get('snippet', '')
        }
    
    return mapping


# Legacy function names for backward compatibility
resolve_urls = create_short_urls
get_citations = lambda response, resolved_urls_map: []  # Simplified for OpenAI