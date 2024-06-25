
from langchain_core.pydantic_v1 import validator
from text_parser import *
from Search_Store import *

class Grader(BaseModel):
    "Use this format to give a binary score for relevance check on retrived documents."

    grade: Literal["relevant", "irrelevant"] = Field(
        ...,
        description="The relevance score for the document.\n"
        "Set this to 'relevant' if the given context is relevant to the user's query, or 'irrlevant' if the document is not relevant.",
    )

    @validator("grade", pre=True)
    def validate_grade(cls, value):
        if value == "not relevant":
            return "irrelevant"
        return value


grader_system_prompt_template = """"You are a grader tasked with assessing the relevance of a given context to a query. 
    If the context is relevant to the query, score it as "relevant". Otherwise, give "irrelevant".
    Do not answer the actual answer, just provide the grade in JSON format with "grade" as the key, without any additional explanation."
    """

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system_prompt_template),
        ("human", "context: {context}\n\nquery: {query}"),
    ]
)


grader_chain = grader_prompt | llm.with_structured_output(Grader, method="json_mode")


rag_template_str = (
    "You are a helpful assistant. Answer the query below based only on the provided context.\n\n"
    "context: {context}\n\n"
    "query: {query}"
)


rag_prompt = ChatPromptTemplate.from_template(rag_template_str)
rag_chain = rag_prompt | llm | StrOutputParser()

# query = "What are the symptoms of malaria?"
# context = retriever.get_relevant_documents(query)

# response = rag_chain.invoke({"query": query, "context": context})

# Markdown(response)


fallback_prompt = ChatPromptTemplate.from_template(
    (
        "You are a friendly medical assistant created by RedxAI.\n"
        "Do not respond to queries that are not related to health.\n"
        "If a query is not related to health, acknowledge your limitations.\n"
        "Provide concise responses to only medically-related queries.\n\n"
        "Current conversations:\n\n{chat_history}\n\n"
        "human: {query}"
    )
)

fallback_chain = (
    {
        "chat_history": lambda x: "\n".join(
            [
                (
                    f"human: {msg.content}"
                    if isinstance(msg, HumanMessage)
                    else f"AI: {msg.content}"
                )
                for msg in x["chat_history"]
            ]
        ),
        "query": itemgetter("query") ,
    }
    | fallback_prompt
    | llm
    | StrOutputParser()
)

fallback_chain.invoke(
    {
        "query": "Hello",
        "chat_history": [],
    }
)
