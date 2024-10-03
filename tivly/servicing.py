from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
import datetime
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END
from .utils import CompleteOrEscalate, State
from tivly.questionaree import search_user_info

# @tool 
# def search_user_info() -> str:
#     "Run this tool to retrieve the customer information"
#     return """Current products:
#                 - Mortgage: 10 Year Closed - 10.2%"""

@tool
def create_ticket(title:str, description:str) -> str: 
    """Run this tool to create a ticket in the CRM for the current issue
    Args: 
        title: str name of the issue
        description: str short description of the issue."""
    
    return 'Ticket {title} successfully created in the CRM with description {description}'

@tool
def lump_sum_is_client_elegible(customer_id:int) -> str: 
    "Run this tool to validate if the client is elegible to make a lump sum payment"
    if customer_id == 0: 
        return "True"
    else:
        return "False"
    
@tool 
def lump_sum_payment_methods() -> str:
    "Run this tool when you need to retrieve the current information for the available payment methods to make a lump sum payment."

    return """Methods of Payment
Online Bill Payment:
Borrower “pays a bill” via own online banking portal
Must be from their bank account and not 3rd party bank account.
Adds the Bank Mortgages as a Payee
Account number is the Bank’s 6-digit mortgage loan number
Amount limitation varies by Financial Institutions
Can submit multiple transactions

Bank Draft/ Personal Cheque:
Personal cheque can be accepted as long as it is from one of borrowers’ bank account.
Drop off/courier to Toronto office at 67 Yonge. Payable to “BL Bank”
"""

@tool 
def send_confirmation_email(email:str) -> str: 
    """Run this tool to send a confirmation notification to the provided email address
    Args: 
        email: str mail address to send notification """
    return "Confirmation email sent to {email}"

lump_sum_prompt = """- Lump sum payment: Borrower(s) are permitted during the term of the mortgage to pay down the mortgage principal in a lump sum. 
The payment during the term must be made within their prepayment privilege to avoid a prepayment charge. 
However, at the time of renewal, borrower(s) are permitted to pay down the principal as much as they want.
            Step 1: Ask for customer ID 
            Step 2: Verify if client is eligibility to make a lump sum payment.
            Step 3: Explain the different payment methods and confirm the clients preference.
            Step 4: I. create a ticket for a human agent to finalize the task with title 'LumpSum'
                    II. send a confirmation email to the client
                    III. send a confirmation email to the payment team at blpayments@gmail.com
"""

# Servicing assistant
servicing_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling banking customer servicing requests. "
            "The primary assistant delegates work to you whenever the user needs help with any of the following kind of servicing issues."
            "As a servicing assistant, you can help customers with inquires related to the following topics: " #Get/fetch client data from CRM
            f"{lump_sum_prompt}"
            """ -Title Change: 
            Step 1: Ask the customers for the change details
            Step 2: create a ticket in the CRM with title 'TitleChange' """
            "Always provide feedback to the custmer when running tools or doing validations."
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            "Remember that a servicing issue isn't completed until after the relevant tool has successfully been used."
            "\n\nCurrent user information:\n\n{user_info}\n"
            # "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.',
        ),
        ("placeholder", "{messages}"),
    ]
)#.partial(time=datetime.now())

servicing_safe_tools = [search_user_info, lump_sum_is_client_elegible, lump_sum_payment_methods, send_confirmation_email]
servicing_sensitive_tools = [create_ticket]
servicing_tools = servicing_safe_tools + servicing_sensitive_tools


class ToServicingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle servicing requests,account updates, lump sums and more."""

    request: str = Field(
        description="Any necessary followup questions the servicing assistant should clarify before proceeding."
    )

def route_servicing(
    state: State,
) -> Literal[
    "servicing_sensitive_tools",
    "servicing_safe_tools",
    # "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in servicing_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "servicing_safe_tools"
    return "servicing_sensitive_tools"


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

    servicing_runnable = servicing_prompt | llm.bind_tools(servicing_tools)
    

    builder = StateGraph(State)

    def user_info(state: State):
        return {"user_info": search_user_info.invoke({})}

    builder.add_node("fetch_user_info", user_info)
    builder.add_edge(START, "fetch_user_info")

    #  renewal assistant assistant
    # builder.add_node(
    #     "enter_renewal",
    #     create_entry_node("Renewal Assistant", "renewal"),
    # )
    builder.add_node("servicing", Assistant(servicing_runnable))
    builder.add_edge("fetch_user_info", "servicing")

    builder.add_node(
        "servicing_sensitive_tools",
        create_tool_node_with_fallback(servicing_sensitive_tools),
    )
    builder.add_node(
        "servicing_safe_tools",
        create_tool_node_with_fallback(servicing_safe_tools),
    )
    builder.add_edge("servicing_sensitive_tools", "servicing")
    builder.add_edge("servicing_safe_tools", "servicing")
    builder.add_conditional_edges("servicing", route_servicing)

    memory = MemorySaver()
    servicing_agent = builder.compile(checkpointer=memory)
 
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
      
        events = servicing_agent.stream(
            {"messages": ("user", _input)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)


