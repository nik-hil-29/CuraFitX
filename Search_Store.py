
from text_parser import *
class VectorStore(BaseModel):
    (
        "A vectorstore contains information about symptoms, treatment"
        ", risk factors and other information about malaria, type 1 and"
        "type 2 diabetes and migraines"
    )

    query: str


class SearchEngine(BaseModel):
    """A search engine for searching other medical information on the web"""

    query: str

class SearchEngine(BaseModel):
    """A search engine for searching other medical information on the web"""

    query: str

router_prompt_template = (
    "You are an expert in routing user queries to either a VectorStore, SearchEngine\n"
    "Use SearchEngine for all other medical queries that are not related to malaria, diabetes, or migraines.\n"
    "The VectorStore contains information on malaria, diabetes, and migraines.\n"
    'Note that if a query is not medically-related, you must output "not medically-related", don\'t try to use any tool.\n\n'
    "query: {query}"
)

llm = ChatGroq(model="Llama3-70b-8192", temperature=0)
prompt = ChatPromptTemplate.from_template(router_prompt_template)
question_router = prompt | llm.bind_tools(tools=[VectorStore, SearchEngine])
