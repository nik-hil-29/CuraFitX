import multiprocessing
if __name__ == '__main__':
    
    # Any other initialization code you might have
    
    # Add freeze_support() here
    multiprocessing.freeze_support()
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import nest_asyncio
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain.chains.combine_documents import stuff
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
import json
import re
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnablePassthrough,
)
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
# import asyncio

import os

GROQ_API_KEY = ""
LANGCHAIN_API_KEY = ""
# LANGCHAIN_PROJECT = user_secrets.get_secret("LANGCHAIN_PROJECT")
TAVILY_API_KEY = ""

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]="Agentic RAG"
os.environ["TAVILY_API_KEY"]=TAVILY_API_KEY

def parse_search_research(results: str):
    pattern = r"\[content: (.*?), title: (.*?), url: (.*?)\]"
    result = re.findall(pattern, results)

    data_list = []
    for snippet, title, link in result:
        data_list.append({"content": snippet, "title": title, "url": link})
    return data_list

urls = [
    "https://www.webmd.com/a-to-z-guides/malaria",
    "https://www.webmd.com/diabetes/type-1-diabetes",
    "https://www.webmd.com/diabetes/type-2-diabetes",
    "https://www.webmd.com/migraines-headaches/migraines-headaches-migraines",
]

loader = WebBaseLoader(urls, bs_get_text_kwargs={"strip": True})
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
chunks = text_splitter.split_documents(docs)

embedding_function = HuggingFaceEmbeddings(show_progress=True, multi_process=True)

vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_function)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
