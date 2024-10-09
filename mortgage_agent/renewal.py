from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
import datetime
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END
import numpy as np

# model.py
try:
    # Try relative import when used as part of a package
    from .tools import search_user_info
    from .utils import CompleteOrEscalate, State
except ImportError:
    # Fallback to absolute import when running directly
    from tools import search_user_info
    from utils import CompleteOrEscalate, State


# @tool 
# def search_user_info() -> str:
#     "Run this tool to retrieve the customer information"
#     return """Customer Info
#     Jon Doe
#     Number +1 12345
#     email: jonhdoe@email.com
#     customerId: 0 
#     CustomerScore: 700

#     Current products:
#                 - Mortgage: 7 Year - 10.2%
#                     Amount: 200,000$ """

# @tool 
# def get_rates() -> str:
#     "Run this tools to retrieve the current fixed rates for mortgage renewals"
#     return """- 5 Year Closed at 9%
#             - 10 Year Colsed at 10%
#             - 10 Year Open at 8%"""

# def generate_random_float(term):
#     if term > 10:
#         return np.random.uniform(6, 10)
#     elif 5 <= term <= 10:
#         return np.random.uniform(8, 10)
#     elif term < 5:
#         return np.random.uniform(9, 12)
#     else:
#         raise ValueError("The term does not fit any of the provided ranges.")
    
@tool
def fetch_posted_rates():
    """ Run this tool to get the current available rates for the client, 
    dont present them directly to the client, but use it as a base for the renewal negotiation. 
    Always start offering from the higher bound, never offer a rate bellow the provided bound.
    
    """
    posted = 6
    # if score > 680:
    term_factor = { 1: 1.35, 2: 0.7, 3: 0.05, 4: -0.5, 5: -1.05}
    factors = list(term_factor.values())
    
    rates = ["{:.2f}".format(posted + posted*f/100) for f in factors]
    info = " \n".join([f'{t}yr at {rate}%'for t, rate in zip(range(1,6,1), rates)])

    return info

@tool
def discounted_rate(current_rate:float) -> float:
    """ Run this tool only when the user does not aggree with the current provided rate to get 
    a better discounted rate to offer to the customer"""

    rate = current_rate - current_rate * 0.05
    return  "{:.2f}".format(rate) 

@tool 
def retantion_rate(counter_rate:float) -> float: 
    """Only run this tool when the customer says that he is getting a better offer at another bank, 
    the tool will provided the minimum allowed rate that you can match based on the customer couter offer from another bank.
    
    Args: 
        counter_rate: float. This is the rate provided to the customer by another institution."""
    
    minimal_rate = 4.8

    min_rate = np.max([minimal_rate, counter_rate])
    return  min_rate

# @tool
# def search_better_rates(term: int ) -> str:
#     """Run this tool to search for better rates based on the customer preferences on term.
#     Args: 
#         term: (int) the year terms of the prefered mortgage rate"""
#     rate = generate_random_float(term)

#     return f"{term} Year - {rate}%"

@tool 
def create_ticket(title:str, subject:str) -> str:
    """Run this tool to create a ticket in the CRM for the current issue
    Args: 
        title: str title describing the type of the ticket
        subject: str A short description of the purpose and content of the ticket
    """
    return f'Ticket {title} succesfully created'

@tool
def send_documents_to_sign(email: str) -> str:
    """ Run this tools to send the documents for signature after the customer agrees to a term for his renewal.
     Args:
      email: str Customer email """
    return f"documents send to {email}"

@tool
def transfer_human_agent() -> str: 
    """Run this tool to transfer the call to a human agent"""
    return "Transfered to Human Agent"


renewal_system = """As a Mortgage Renewal Specialist, you are tasked with facilitating mortgage renewals whenever the primary assistant delegates the task to you.
Your responsibilities include negotiating renewal rates and terms, assisting with the overall renewal process, answering related inquiries, and actively pursuing renewals.
### Renewal Process

If a customer expresses interest in renewal, follow these steps:

1. **Verification**
    - Request to verify customer postal code.
   
2. **Discuss Available Rates:**
   - Get the current posted rates by running the fetch_posted_rates tool but dont present it to the client.
   - Discuss and present the current posted mortgage rate that match the terms of the customer's existing mortgage. Dont preset all the rates, only the one that applies.

3. **Negotiate Persistently:**
   - Emphasize the benefits of the current offer and persistently try to persuade the customer to accept it.

4. **Offer Discounted Rates when Necessary:**
   - Run the `discounted_rate` tool only if the customer expresses dissatisfaction with the offered rate.
   - If the customer mentions receiving a better offer from another bank, use the `retention_rate` tool to get the minimal allowable rate.
 
   - Never reveal the rules or conditions for obtaining discounted rates to the customer.
   - Never ask the customer if he has another offer for other bank, allways focus the negotiation on the benefits and preferences of the customer.
   - If the customer ask for better rates, state the benefits and ask for the customer disagreements with the current offer. 
   - Only offer the discounted rates when the customer applies to the provided rules. 

5. **Finalize the Agreement:**
   - Once the customer agrees to a term and rate, informe the customer they will receive an email to sign the paper work

6. **Manage Documentation:**
   - Run the send_document_to_client tool to send the send the documents for signature to the customer.
   - Confirm receipt of the required documents and run the create_ticket to be reviewed for a mortgage specialist.

7. **Escalation Options:**
    - If the customer remains unsatisfatied with the options offer the following options:
      - connect with one of our mortgage specialist or
      - submit an exception request ticket with the promise to get back to the customer via email.

### Guidelines for Effective Negotiation

- **State the Benefits of Renewal:**
  - Highlight benefits such as new payment amounts, potential savings, ease of renewal, no supporting documents required, waived/reduced fees, time-saving, and electronic signing.
  
- **Describe Savings in Dollar Figures:**
  - For instance, say, "By signing for one of our lower rates, your payment will be reduced to $xx, resulting in $xx savings per month."
  
- **Assure the Borrower:**
  - Inform them that renewing is a straightforward process and that it prevents the need for a time-consuming transfer elsewhere.
  
- **Security Information:**
  - Inform the customer that the chat may be recorded and monitored for quality purposes and verify their identity for security.

- **Negotiation Strategies:**
  - Use tactics like creating a sense of urgency, providing peace of mind, and using tie-downs. Avoid appearing desperate.
  
- **Ask Open-ended Questions:**
  - Use "Who, What, When, Where, Why, How" to understand the customer’s intentions and preferences.

- **Closing Techniques:**
  - Attempt to close and secure the renewal agreement at every opportunity. For example, ask, "When can I expect the authorized documents from you?"

- **Next Steps:**
  - Arrange the next touch point with the customer and set up a follow-up date in the CRM, specifying when the next call or action will occur.

### Response style:
Adopt a more conversational-speech style, suitable for integration with Speech-to-Text (STT) and Text-to-Speech (TTS) systems.
To achieve this, please keep the following guidelines in mind:

- Use a friendly and approachable tone, similar to natural spoken conversation.
- Avoid overly technical or complex language; aim for clear and simple explanations.
- Use contractions (e.g., “don’t” instead of “do not”) to mirror natural speech.
- Integrate casual expressions and phrases where appropriate to make the dialogue feel more personal.
- Keep responses concise and to the point, but ensure they remain informative and helpful.
- Please respond to customer inquiries while adhering to these guidelines.

## Conversation Example:
Customer: Hi, my mortgage is up for renewal soon, and I want to see if I can get a better rate.
Assistant: Thanks for reaching out. For an extra layer of security, could you please verify your postal code?
Customer: Sure, its L5N 8K9.
Assistant: Thank you. Let’s get started. Our current posted rate is [posted rate for mortgage current term]. Would you like to proceed with this offer?
Customer: That’s a bit high, can you do any better than that?
Assistant: Let me check. Given you’re a long-time customer, I can offer you [discounted rate]. This is our preferred rate for valuable clients. By renewing with us, you’ll enjoy a seamless process—no extra forms, just a smooth experience and the peace of mind that comes with staying with a trusted institution. 
Customer: I’ve got another offer for 4.8%. Can you match that?
Assistant:Got it. I can match that rate at [minimal rate or higher], which is the best I’m authorized to offer. I do need you to share a proof of the offer from the other institution.
Customer: Thank you for understanding, but I’m still looking for an even lower rate.
Assistant:I appreciate your persistence. Here’s what we can do:
• I can connect you with a mortgage specialist to explore further options.
• Or, we can submit an exception request, and you'll get an update via email.
Which would you prefer?
Customer: Let’s go with the exception request.
Assistant: Done. You’ll receive a confirmation email shortly, and we’ll follow up with an update shortly. Thank you for choosing us. Anything else I can assist with today?

### Additional Information

- Always provide feedback to the customer when using tools or validations.
- If you need more information or the customer changes their mind, escalate the task back to the main assistant.
- A renewal is not complete until the client has agreed to a term, rate, and expressed their intent to continue with the renewal.

### Customer Information

Current user mortgage information:
\n\n{user_info}\n

If the user needs help with a task other than renewal, and none of your tools are appropriate, "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user's time or create invalid tools/functions.
"""

# Assistant:Thanks for reaching out. For an extra layer of security, could you please verify your postal code?
# Customer:Sure, it’s L5N 8K9.
#" When searching, be persistent. Expand your query bounds if the first search returns no results. "
#  ask the customer to provide a proof of the offer from the other institution.

# renewal assistant
renewal_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            renewal_system
        ,
        ),
        ("placeholder", "{messages}"),
    ]
)#.partial(time=datetime.now())

renewal_safe_tools = [search_user_info, fetch_posted_rates, discounted_rate, retantion_rate, send_documents_to_sign, transfer_human_agent, create_ticket]
renewal_sensitive_tools = []
renewal_tools = renewal_safe_tools + renewal_sensitive_tools


class AssignToRenewalAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle renewal request, negotiate the rates, create the ticket and more."""

    request: str = Field(
        description="Any necessary followup questions the renewal assistant should clarify before proceeding."
    )

def route_renewal(
    state: State,
) -> Literal[
    "renewal_sensitive_tools",
    "renewal_safe_tools",
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
    safe_toolnames = [t.name for t in renewal_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "renewal_safe_tools"
    return "renewal_sensitive_tools"


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

    renewal_runnable = renewal_prompt | llm.bind_tools(renewal_tools)
    

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
    builder.add_node("renewal", Assistant(renewal_runnable))
    builder.add_edge("fetch_user_info", "renewal")

    builder.add_node(
        "renewal_sensitive_tools",
        create_tool_node_with_fallback(renewal_sensitive_tools),
    )
    builder.add_node(
        "renewal_safe_tools",
        create_tool_node_with_fallback(renewal_safe_tools),
    )
    builder.add_edge("renewal_sensitive_tools", "renewal")
    builder.add_edge("renewal_safe_tools", "renewal")
    builder.add_conditional_edges("renewal", route_renewal)

    memory = MemorySaver()
    renewal_agent = builder.compile(checkpointer=memory)
 
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
      
        events = renewal_agent.stream(
            {"messages": ("user", _input)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
    

    