# CuraFitX
In this Project, I built a RAG (Retrieval Augmented Generation) system to answer medical queries. The agent's primary duty is to fetch chunks from a vector store to answer a user's query. However, the retrieved chunks are not just given as-is to the language model (LLM). They are first scored, and only the chunks that are relevant to the query are passed to the LLM.

In cases where all the retrieved chunks are not relevant to the query, the agent performs a web search. The agent also performs a web search for queries that do not have any relevant documents in the vector store.

Before the LLM's response is presented to the user, the system performs a hallucination and answer relevance check. Lastly, if the query is not related to health, the system falls back to a different chain to respond based on its knowledge.

Stack Used:

- Chromadb
- Tavily AI
- HuggingFaceHubEmbeddings (sentence-transformers/all-mpnet-base-v2)
- Llama 3 70B from Groq
- Langchain and LangGraph
- Gradio
You can checkout this in my running space on Hugging Face Here : https://huggingface.co/spaces/nikhil2902/CuraFitX
