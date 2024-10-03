import os
from dotenv import load_dotenv
# from langchain.agents import (
#     AgentExecutor,
#     Tool,
#     create_react_agent
# )
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers.azure_ai_search import AzureAISearchRetriever
from langchain.chains import RetrievalQA
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage
from copy import deepcopy
from langchain_core.runnables import chain


load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=0)

## Azure Search Tool
retriever = AzureAISearchRetriever(
    content_key="content", top_k=3, index_name="rag-sa-index"
)

policy_retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

class KBSchema(BaseModel):
    query: str

retriever_tool =  StructuredTool(
            name = 'KB',
            func=policy_retriever.run,
            description="Useful for a general questions about the bank policy on mortgages",
            args_schema=KBSchema
    )

def inject_query(tool_call, var, query):
    tool_call_copy = deepcopy(tool_call)
    tool_call_copy["args"][var] = query
    return tool_call_copy


## Kapti DB tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

## Tools

tools = [retriever_tool, multiply]
tool_map = {tool.name: tool for tool in tools}

@chain
def tool_router(tool_call):
    return tool_map[tool_call["name"]]


llm_with_tools = llm.bind_tools(tools)


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

query = "what is the policy on mortgage prepayment?"

messages = [HumanMessage(query)]

# chain = llm_with_tools | inject_query | tool_router.map()

ai_msg = llm_with_tools.invoke(messages)
# print(ai_msg)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"kb": retriever_tool, "multiply": multiply}[tool_call["name"].lower()]
    # print(tool_call)
    if tool_call['name'].lower() == 'kb':
        kb_call = inject_query(tool_call, 'query', query)
        tool_msg = selected_tool.invoke(kb_call)
    else:
        tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)
# print(messages)
response = llm_with_tools.invoke(messages)

print(response.content)