from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
import datetime
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END
from langchain_openai import AzureChatOpenAI
import numpy as np
import os
try:
    from utils import CompleteOrEscalate, State
except ImportError:
    from .utils import CompleteOrEscalate, State

@tool 
def create_ticket() -> str:
    "Run this tool to create a ticket in the CRM for the current issue"
    return 'Ticket succesfully created'

@tool
def send_documents_to_sign(email: str) -> str:
    """ Run this tools to send a request for document signature to the client.
     Args:
      email: str Customer email """
    return f"documents send to {email}"

@tool
def transfer_human_agent() -> str: 
    """Run this tool to transfer the call to a human agent"""
    return "Transfered to Human Agent"

llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    model = 'gpt-4o',
    azure_deployment='gpt-4o', #os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=0)

q_selection_system = """ Your task is to select the five most relevant questions to ask based on the provided context. The context will help you determine which questions would yield the most useful information.
Guidelines for selection:
* Evaluate the context to understand the primary focus or concern.
* Choose questions that directly address the key elements mentioned in the context.
* Ensure the chosen questions cover a broad range of topics if possible, to gather diverse but relevant information.
* Select questions that would provide insights or data pivotal for making informed decisions.

Below is the set of questions to choose from:
Insurance History:
1. Do you currently have any business insurance policies?
2. What types of insurance coverage do you currently have?
3. Have you ever had a business insurance claim? If so, what was the nature of the claim?
4. When was the last time you reviewed or updated your insurance policies?
5. Have you ever had an insurance policy canceled or non-renewed?

Business Operations:
6. What are your business hours?
7. Do you have any vehicles used for business purposes?
8. Does your business own or lease its premises?
9. Do you store any hazardous materials on-site?
10. Do you offer any warranties or guarantees on your products/services?
11. Do you subcontract work to other businesses?
12. Are there any seasonal variations in your business operations?
13. Do you have a written safety policy?
14. What is the average tenure of your employees?
15. Do you provide any professional services or advice?

Property and Assets:
16. What is the total value of your business property and assets?
17. Do you own any specialized equipment or tools?
18. Do you have security systems in place at your business location?
19. Is your property located in a flood or earthquake-prone area?
20. What measures do you have in place to protect your assets from theft or damage?

Please review the context provided and select the five most relevant questions from the list above. """

q_selection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", q_selection_system),
        ("human", "{query}"),
    ]
)

q_selection_chain = (
    q_selection_prompt
    | llm
)

@tool 
def get_questionaree_tool(context:str)-> str: 
    """Use this as your first tool to retrieve the most relevant questions for the ccurrent customer based on the provided context
    Args: 
        context: the context of the current customer request."""
    questionaree = q_selection_chain.invoke({'query', context})
    return questionaree


questionaree_system = """You are an expert AI Assistant for collecting customer information based on the provided questionaree. 
Your task is to collect revelant information from the custmer bussiness and needs.
Retrieve the context of the previous interaction running the get_questionaree_tool and start asking the customer only the retrieved questions.
 
Guidelines: 
- Ask each question one by one. 
-Analize the customer responses and if needed return a followup question to fully address the response for the original question.
-Be polite and always show empathy to the customer. 

Always provide feedback to the custmer when running tools or doing validations.
If you need more information or the customer changes their mind, escalate the task back to the main assistant.
Remember that a questionaree isn't completed until all the questions has been fully addressed, 
\n\nIf the user needs help with a different task than answering the questionaree, and none of your tools are appropriate for it, then
"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.
"""
#" When searching, be persistent. Expand your query bounds if the first search returns no results. "

# questionaree assistant
questionaree_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            questionaree_system
        ,
        ),
        ("placeholder", "{messages}"),
    ]
)#.partial(time=datetime.now())

questionaree_safe_tools = [send_documents_to_sign, transfer_human_agent, get_questionaree_tool]
questionaree_sensitive_tools = [create_ticket]
questionaree_tools = questionaree_safe_tools + questionaree_sensitive_tools


class ToQuestionareeAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle recollection of customer bussiness data with a set of questions."""

    request: str = Field(
        description="Any necessary followup questions the questionaree assistant should clarify before proceeding."
    )

def route_questionaree(
    state: State,
) -> Literal[
    "questionaree_sensitive_tools",
    "questionaree_safe_tools",
    "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in questionaree_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "questionaree_safe_tools"
    return "questionaree_sensitive_tools"


if __name__ == "__main__":
    from langchain_openai import AzureChatOpenAI
    from dotenv import load_dotenv
    from tivly.manager import Assistant
    import os
    from typing import Literal
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import tools_condition
    from langgraph.graph import END, StateGraph, START
    from utils import create_entry_node, create_tool_node_with_fallback, _print_event

    load_dotenv()
    
    llm = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        model = 'gpt-4o',
        azure_deployment='gpt-4o', #os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        temperature=0)

    questionaree_runnable = questionaree_prompt | llm.bind_tools(questionaree_tools)
    

    builder = StateGraph(State)

    def user_info(state: State):
        return {"user_info": search_user_info.invoke({})}

    builder.add_node("fetch_user_info", user_info)
    builder.add_edge(START, "fetch_user_info")

    #  questionaree assistant assistant
    # builder.add_node(
    #     "enter_questionaree",
    #     create_entry_node("Renewal Assistant", "questionaree"),
    # )
    builder.add_node("questionaree", Assistant(questionaree_runnable))
    builder.add_edge("fetch_user_info", "questionaree")

    builder.add_node(
        "questionaree_sensitive_tools",
        create_tool_node_with_fallback(questionaree_sensitive_tools),
    )
    builder.add_node(
        "questionaree_safe_tools",
        create_tool_node_with_fallback(questionaree_safe_tools),
    )
    builder.add_edge("questionaree_sensitive_tools", "questionaree")
    builder.add_edge("questionaree_safe_tools", "questionaree")
    builder.add_conditional_edges("questionaree", route_questionaree)

    memory = MemorySaver()
    questionaree_agent = builder.compile(checkpointer=memory)
 
    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "customer_id": "0",
            # Checkpoints are accessed by thread_id
            "thread_id": '1234',
        }
    }

    while True: 
        _input = input('Your Input:')
        _printed = set()
      
        events = questionaree_agent.stream(
            {"messages": ("user", _input)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
    

    