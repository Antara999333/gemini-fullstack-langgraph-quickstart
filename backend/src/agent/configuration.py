import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="gpt-4o-mini",
        metadata={
            "description": "The OpenAI model for query generation. Using gpt-4o-mini for cost efficiency."
        },
    )

    reflection_model: str = Field(
        default="gpt-4o-mini",
        metadata={
            "description": "The OpenAI model for reflection and gap analysis. Using gpt-4o-mini for cost efficiency."
        },
    )

    answer_model: str = Field(
        default="gpt-4o",
        metadata={
            "description": "The OpenAI model for final answer synthesis. Using gpt-4o for best quality."
        },
    )

    search_model: str = Field(
        default="gpt-4o-mini",
        metadata={
            "description": "The OpenAI model for web research analysis. Using gpt-4o-mini for cost efficiency."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={
            "description": "The number of initial search queries to generate. Recommended: 2-5."
        },
    )

    max_research_loops: int = Field(
        default=2,
        metadata={
            "description": "The maximum number of research loops to perform. Recommended: 1-3."
        },
    )

    search_results_per_query: int = Field(
        default=8,
        metadata={
            "description": "The number of search results to retrieve per query. Recommended: 5-10."
        },
    )

    max_tokens_per_model: dict = Field(
        default={
            "gpt-4o-mini": 4096,
            "gpt-4o": 8192,
            "gpt-3.5-turbo": 4096,
        },
        metadata={
            "description": "Maximum tokens for different models to prevent truncation."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
    
    def get_model_max_tokens(self, model_name: str) -> int:
        """Get the maximum tokens for a given model."""
        return self.max_tokens_per_model.get(model_name, 4096)