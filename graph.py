from typing import Union, List, Dict

from Grader import *
from Answer_grader import *
from tools import *
from Hallucination_grader import *

def retrieve_node(state: Dict[str, Union[str, List[Document]]]) -> Dict[str, Union[str, List[Document]]]:
    """
    Retrieve relevant documents from the vectorstore.

    Args:
        state (dict): A dictionary containing the query string under the key "query".

    Returns:
        dict: A dictionary containing a key "documents" with a list of retrieved Document objects.
    """
    query = state["query"]
    documents = retriever.invoke(input=query)
    return {"documents": documents}



def fallback_node(state: dict):
    """
    Fallback to this node when there is no tool call
    """
    query = state["query"]
    chat_history = state["chat_history"]
    generation = fallback_chain.invoke({"query": query, "chat_history": chat_history})
    return {"generation": generation}


def filter_documents_node(state: dict):
    filtered_docs = list()

    query = state["query"]
    documents = state["documents"]
    for i, doc in enumerate(documents, start=1):
        grade = grader_chain.invoke({"query": query, "context": doc})
        if grade.grade == "relevant":
            print(f"---CHUCK {i}: RELEVANT---")
            filtered_docs.append(doc)
        else:
            print(f"---CHUCK {i}: NOT RELEVANT---")
    return {"documents": filtered_docs}


def rag_node(state: dict):
    query = state["query"]
    documents = state["documents"]

    generation = rag_chain.invoke({"query": query, "context": documents})
    return {"generation": generation}


def web_search_node(state: dict):
    query = state["query"]
    results = tavily_search.invoke(query)
    # results = parse_search_research(results)
    documents = [
        Document(page_content=doc["content"], metadata={"source": doc["url"]})
        for doc in results
    ]
    return {"documents": documents}


def question_router_node(state: dict):
    query = state["query"]
    try:
        response = question_router.invoke({"query": query})
    except Exception:
        return "llm_fallback"

    if "tool_calls" not in response.additional_kwargs:
        print("---No tool called---")
        return "llm_fallback"

    if len(response.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide route!"

    route = response.additional_kwargs["tool_calls"][0]["function"]["name"]
    if route == "VectorStore":
        print("---Routing to VectorStore---")
        return "VectorStore"
    elif route == "SearchEngine":
        print("---Routing to SearchEngine---")
        return "SearchEngine"


def should_generate(state: dict):
    filtered_docs = state["documents"]

    if not filtered_docs:
        print("---All retrived documents not relevant---")
        return "SearchEngine"
    else:
        print("---Some retrived documents are relevant---")
        return "generate"


def hallucination_and_answer_relevance_check(state: dict):
    llm_response = state["generation"]
    documents = state["documents"]
    query = state["query"]

    hallucination_grade = hallucination_grader_chain.invoke(
        {"response": llm_response, "context": documents}
    )
    if hallucination_grade.grade == "no":
        print("---Hallucination check passed---")
        answer_relevance_grade = answer_grader_chain.invoke(
            {"response": llm_response, "query": query}
        )
        if answer_relevance_grade.grade == "yes":
            print("---Answer is relevant to question---\n")
            return "useful"
        else:
            print("---Answer is not relevant to question---")
            return "not useful"
    print("---Hallucination check failed---")
    return "generate"

from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentSate)
workflow.add_node("VectorStore", retrieve_node)
workflow.add_node("SearchEngine", web_search_node)
workflow.add_node("filter_docs", filter_documents_node)
workflow.add_node("fallback", fallback_node)
workflow.add_node("rag", rag_node)

workflow.set_conditional_entry_point(
    question_router_node,
    {
        "llm_fallback": "fallback",
        "VectorStore": "VectorStore",
        "SearchEngine": "SearchEngine",
    },
)

workflow.add_edge("VectorStore", "filter_docs")
workflow.add_edge("SearchEngine", "filter_docs")
workflow.add_conditional_edges(
    "filter_docs", should_generate, {"SearchEngine": "SearchEngine", "generate": "rag"}
)
workflow.add_conditional_edges(
    "rag",
    hallucination_and_answer_relevance_check,
    {"useful": END, "not useful": "SearchEngine", "generate": "rag"},
)

workflow.add_edge("fallback", END)

# %%
app = workflow.compile(debug=False)
