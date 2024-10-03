import os
from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
    Tool,
    create_react_agent
)
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers.azure_ai_search import AzureAISearchRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool
from db_settings import connect_db
from db_operations import get_customer_all_data
from langchain_core.tools import InjectedToolArg, tool
from typing import Annotated, List

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=0)

## Azure Search Tool
retriever = AzureAISearchRetriever(
    content_key="content", top_k=3, index_name="rag-sa-index"
)

# policy_retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


retriever_tool =  Tool(
        name = 'Knowledge Base',
        func="policy_retriever.run",
        description="Useful for general questions about the bank policy on mortgages"
    )


## Kapti DB tool

conn = connect_db()

@tool
def customer_info(customer: Annotated[dict, InjectedToolArg]
) -> str:
    """Use this tool when you need to retrieve the customer information including previous conversations, products and more.
    
    Args:
    customer: dict with customer id and contact id."""

    data = get_customer_all_data(conn, customer.id, customer.contact_id)
    return data

## Agent
tools = [retriever_tool, customer_info]


template_with_history = """You are a servicing assistant, your task is to provides informative answers to users.
Answer the following questions as best you can. 
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to give detailed, informative answers

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate(template=template_with_history, 
                        input_variables=["input", "intermediate_steps","tools", "tool_names", "history"])

multi_tool_names = [tool.name for tool in tools]
agent = create_react_agent(llm,
                           tools= tools,
                           prompt= prompt)

memory = ConversationBufferWindowMemory(k=2)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

res = agent_executor.invoke({"input":"what is 10*10 rootsquare 10"})
print(res)
