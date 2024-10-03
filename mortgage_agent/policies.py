
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
from dotenv import load_dotenv
load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=0)

## RAG Policy
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
        ("human", "{chat_history}"), # MessagesPlaceholder("chat_history"),
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
        ("human", "{chat_history}"),# MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

@tool
def rag_tool(query:str, 
             chat_history:str)-> str:
    """Use this tool answer questions and retrieve information related to mortgage policies like renewal, lumpsum, and more. 
    
    Args:
    query: (str) the user question
    chat_history: (str) list of previous interactions used to contextualize the final user question"""
    response = rag_chain.invoke({'input': query, 'chat_history': chat_history})
    return response["answer"]

class KBSchema(BaseModel):
    query: str

# def inject_query(tool_call, var, query):
#     tool_call_copy = deepcopy(tool_call)
#     tool_call_copy["args"][var] = query
#     return tool_call_copy