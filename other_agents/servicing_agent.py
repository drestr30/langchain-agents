import os
from dotenv import load_dotenv

from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers.azure_ai_search import AzureAISearchRetriever
# from langchain.chains import RetrievalQA
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from copy import deepcopy
from langchain_core.runnables import chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import InjectedToolArg, tool
from typing import Annotated, List
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging

# load_dotenv()
# conn = connect_db()
# read_prompt_template(conn)


llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=0)

### RAG Policy
#Azure Search Tool
retriever = AzureAISearchRetriever(service_name= os.environ["AZURE_SEARCH_SERVICE"],
                                    api_key= os.environ["AZURE_SEARCH_API_KEY"],
                                    index_name=os.environ["AZURE_SEARCH_INDEX"],
                                    content_key="content", top_k=3)

# System for rag_chain
rag_system_prompt = (
    """ You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say that you don't know. 
    Use three sentences maximum and keep the answer concise."""
    "\n\n"
    "{context}"
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Incorporate the retriever into a question-answering chain.
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

@tool
def rag_tool(query:Annotated[str, InjectedToolArg], 
             chat_history:Annotated[list, InjectedToolArg])-> str:
    """Use this tool answer questions and retrieve information related to mortgage policies like renewal, lumpsum, and more. 
    
    Args:
    query: the user question"""
    response = rag_chain.invoke({'input': query, 'chat_history': chat_history})
    return response["answer"]

class KBSchema(BaseModel):
    query: str

def inject_query(tool_call, var, query):
    tool_call_copy = deepcopy(tool_call)
    tool_call_copy["args"][var] = query
    return tool_call_copy

### Kapti DB tool

@tool
def customer_tool(customer: Annotated[dict, InjectedToolArg]
) -> str:
    """Use this tool to retrieve the customer information including previous conversations, current owned products and more.
    
    Args:
    customer: dict with customer id and contact id."""

    # data = get_customer_all_data(conn, customer["id"], customer["contact_id"])
    # return data

## Tools 

tools = [rag_tool, customer_tool]
tool_map = {tool.name: tool for tool in tools}

def tool_router(tool_call):
    return tool_map[tool_call["name"].lower()]


llm_with_tools = llm.bind_tools(tools)

### Agent prompt 

agent_system_prompt = (
"""You are an expert servicing agent the helps the customer inquiries, your task is to provide information for the user's questions about different topics related to the customer banking products, policies, processes, and more.
Always use the provided tools to provide relevant and truthful responses.  
Never answer a question with information not provided by a tool.  
If the context provided from the tools is not enough to answer the question, clarify that you are not sure about the answer."""
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_system_prompt),
        ("human", "{input}")
        ]
)

agent_llm = prompt | llm_with_tools 

class QnAAgent():

    def invoke(self, history, customer):
        logging.info('Invoke QnA Agent')
        messages = self.history_to_messages(history)
        query = messages[-1].content
        chat_history = messages[:-1]
          # print('history:', chat_history)
        ai_msg = agent_llm.invoke(messages)
        logging.info('AI Msg: ' + str(ai_msg))

        if not ai_msg.tool_calls: #Answer directly if not need to call tools.
            return ai_msg.content
        
        messages.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            selected_tool = tool_router(tool_call)
            # print(tool_call)
            if tool_call['name'].lower() == 'rag_tool':
                kb_call = inject_query(tool_call, 'query', query)
                kb_call = inject_query(kb_call, 'chat_history', chat_history)
                tool_msg = selected_tool.invoke(kb_call)
                logging.info('Tool message:' + str(tool_msg))
            else:
                customer_info_call = inject_query(tool_call, 'customer', customer)
                tool_msg = selected_tool.invoke(customer_info_call)
                logging.info('Tool message: Customer Info Tool Executed')
            
            messages.append(tool_msg)
        response = agent_llm.invoke(messages)

        logging.info('LLM Response' + str(response))

        return response.content

    def history_to_messages(self, history):
        messages = []
        for utter in history:
            if utter['role'] == 'user':
                messages.append(HumanMessage(utter['content']))
            elif utter['role'] == 'assistant':
                messages.append(AIMessage(utter['content']))
        return messages
    
qna_agent = QnAAgent()


if __name__ == "__main__":
    # query = "what is the policy on mortgage renwal?"

    
    chat_history = []
    query = input('Input:')
    messages = [HumanMessage(query)]
    chat_history = [HumanMessage(query)]

    while True:
        # print('history:', chat_history)
        ai_msg = agent_llm.invoke(messages)
        messages.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            selected_tool = tool_router(tool_call)
            # print(tool_call)
            if tool_call['name'].lower() == 'rag_tool':
                kb_call = inject_query(tool_call, 'query', query)
                kb_call = inject_query(kb_call, 'chat_history', chat_history)
                tool_msg = selected_tool.invoke(kb_call)
            else:
                id = 1005
                customer_info_call = inject_query(tool_call, 'customer', {'id':1005, 'contact_id':17438})
                tool_msg = selected_tool.invoke(customer_info_call)
            messages.append(tool_msg)

        print(messages)
        response = agent_llm.invoke(messages)

        print('----------------------')
        print("Response: ", response.content)
        chat_history.append(AIMessage(response.content))

        query = input('Input:')
        chat_history.append(HumanMessage(query))
        messages = chat_history.copy()

    # response = qna_agent.invoke([{'role': 'user', 'content': 'policy on renewal'}], {'id': 1005, 'contact_id': 17977})
    # print(response)
