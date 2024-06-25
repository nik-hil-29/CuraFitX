from text_parser import *
from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.tools import Tool
from langchain_core.messages.base import BaseMessage
import operator

# ddg_search = DuckDuckGoSearchResults()
tavily_search = TavilySearchResults()
tool_executor = ToolExecutor(
    tools=[
        Tool(
            name="VectorStore",
            func=retriever.invoke,
            description="Useful to search the vector database",
        ),
        Tool(
            name="SearchEngine", func=tavily_search, description="Useful to search the web"
        ),
    ]
)


class AgentSate(TypedDict):
    """The dictionary keeps track of the data required by the various nodes in the graph"""

    query: str
    chat_history:list[BaseMessage]
    generation: str
    documents: list[Document]

