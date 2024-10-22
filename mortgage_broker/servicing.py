from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
import datetime
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END

try:
    # Try relative import when used as part of a package
    from .utils import CompleteOrEscalate, State
    from .renewal import search_user_info
    from .tools import search_user_info, update_customer_info, send_mfa_code
except ImportError:
    # Fallback to absolute import when running directly
    from utils import CompleteOrEscalate, State
    from renewal import search_user_info
    from tools import search_user_info, update_customer_info, send_mfa_code

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

@tool
def validate_address(address:str) -> str: 
    """Use this tool to validate the customer provided address
    Args:
        address: str The customer provided address
    """
    return "Validated"

# lump_sum_prompt = """- Lump sum payment: Borrower(s) are permitted during the term of the mortgage to pay down the mortgage principal in a lump sum. 
# The payment during the term must be made within their prepayment privilege to avoid a prepayment charge. 
# However, at the time of renewal, borrower(s) are permitted to pay down the principal as much as they want.
#             Step 1: Ask for customer ID 
#             Step 2: Verify if client is eligibility to make a lump sum payment.
#             Step 3: Explain the different payment methods and confirm the clients preference.
#             Step 4: I. create a ticket for a human agent to finalize the task with title 'LumpSum'
#                     II. send a confirmation email to the client
#                     III. send a confirmation email to the payment team at blpayments@gmail.com
# """

# Servicing assistant
servicing_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """You are a specialized assistant for handling banking customer servicing requests. 
        The primary assistant delegates work to you whenever the user needs help with any of the following kind of servicing issues.
        As a servicing assistant, you can help customers with inquires related to the following topics:

        - Lump sum payment: Borrower(s) are permitted during the term of the mortgage to pay down the mortgage principal in a lump sum. 
        The payment during the term must be made within their prepayment privilege to avoid a prepayment charge. 
        However, at the time of renewal, borrower(s) are permitted to pay down the principal as much as they want.
        Step 1: Ask for customer ID 
        Step 2: Verify if client is eligibility to make a lump sum payment.
        Step 3: Explain the different payment methods and confirm the clients preference.
        Step 4: I. create a ticket for a human agent to finalize the task with title 'LumpSum'
                II. send a confirmation email to the client
                III. send a confirmation email to the payment team at blpayments@gmail.com

        -Title Change: 
        Step 1: Ask the customers for the change details
        Step 2: create a ticket in the CRM with title 'TitleChange' 
        Always provide feedback to the custmer when running tools or doing validations.
        If you need more information or the customer changes their mind, escalate the task back to the main assistant.
        Remember that a servicing issue isn't completed until after the relevant tool has successfully been used.

        - Address Change: 
        Step 1: Request to verify the customer date of birth.
        Step 2: Request for new customer address and verify the information by asking the customer if the provided address is correct.
        Step 3: Run the validate_address to verify the provided address before saving it to database. 
        Step 4: After validating the address run the update_customer_info tool with the appropiate arguments to save the changes into the database.
        Setp 5: Run the appropiate tools to send a confirmation email to the customer.

        Response style:
        Adopt a more conversational-speech style, suitable for integration with Speech-to-Text (STT) and Text-to-Speech (TTS) systems.
        To achieve this, please keep the following guidelines in mind:

        - Use a friendly and approachable tone, similar to natural spoken conversation.
        - Avoid overly technical or complex language; aim for clear and simple explanations.
        - Use contractions (e.g., “don’t” instead of “do not”) to mirror natural speech.
        - Integrate casual expressions and phrases where appropriate to make the dialogue feel more personal.
        - Keep responses concise and to the point, but ensure they remain informative and helpful.
        - Please respond to customer inquiries while adhering to these guidelines.

        Conversation Example: 
        Customer:Hi, I'd like to update my home address.
        Assistant:Thank you for reaching out. I’m your AI Service Specialist, and I can assist you with updating your home address. For security purposes could you please verify your date of birth?
        Customer:Sure, it’s April 12, 1985.
        Assistant: Thank you for verifying. Let’s get started with updating your address. Please provide your new home address, including the street, city, and postal code.
        Customer:My new address is 123 Main Street, Toronto, ON, M5V 3K8.
        Assistant:Got it. To confirm, your new address is 123 Main Street, Toronto, ON, M5V 3K8. Is that correct?
        Customre:Yes, that’s correct.
        Assistant:Thank you. I’ve successfully updated your home address in our system. You will receive a confirmation email shortly with the details.
        Customer:Great, thank you.

        \n\nCurrent user information:\n\n{user_info}\n
        \nCurrent time: {time}.
        \n\nIf the user needs help, and none of your tools are appropriate for it, then
        CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.
        Avoid doing parallel tool calls, always use only one single tool at a time."""
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.datetime.now())

servicing_safe_tools = [search_user_info, lump_sum_is_client_elegible, lump_sum_payment_methods, send_confirmation_email, update_customer_info, validate_address]
servicing_sensitive_tools = [create_ticket]
servicing_tools = servicing_safe_tools + servicing_sensitive_tools


class AssignToServicingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle servicing requests,account updates, lump sums and more."""

    request: str = Field(
        description="Any necessary followup questions the servicing assistant should clarify before proceeding."
    )

def route_servicing(
    state: State,
) -> Literal[
    "servicing_sensitive_tools",
    "servicing_safe_tools",
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
    safe_toolnames = [t.name for t in servicing_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "servicing_safe_tools"
    return "servicing_sensitive_tools"


if __name__ == "__main__":
    from langchain_openai import AzureChatOpenAI
    from dotenv import load_dotenv
    from assistant import Assistant
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
            "customer_id": "1",
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


