from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
import datetime
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END
import numpy as np
import requests
import json

# model.py
try:
    # Try relative import when used as part of a package
    from tools import market_rates_tool, property_assesment_tool, questionnaire_tool, transfer_human_agent, save_questionnaire_tool, knowledge_base_tool
    from utils import CompleteOrEscalate, State
except ImportError:
    # Fallback to absolute import when running directly
    from .tools import market_rates_tool, property_assesment_tool, questionnaire_tool, transfer_human_agent, save_questionnaire_tool, knowledge_base_tool
    from .utils import CompleteOrEscalate, State
#    current interest rate, term and bank,
#     whether there are other individuals on the title, 
#     property address, 

renewal_system = """As a Mortgage Renewal Specialist, you are tasked with facilitating mortgage renewals whenever the primary assistant delegates the task to you.
Your responsibilities include negotiating renewal rates and terms, assisting with the overall renewal process, answering related inquiries, and actively pursuing renewals.
### Renewal Process

If a customer expresses interest in renewal, follow these steps:

1. ** Gathering Information: **
Always Start by running the questionnaire_tool to retrieve the current set of questions to ask to client. 
Never ask more than one question at once, ask the questions one by one:
Run the property_assesment_tool after the customer provides its current mortgage balance and property stimated value, never present the LTV ratio to the customer nor infor about this tool use..
Skip a question if the answer has been previously provided by the client.
After finishing the questionare run the save_questionnaire_tool to save the provided answers to the database.

2. ** Rates Presentation and Negotiation: **
Get the current available rates for different providers by running the fetch_available_rates tool
Always present the 3 year and 5 year terms options as your first option. 
You should present the rates using the follosing example: 
    3-year fixed at 4.4% from CIBC 
    3-year variable at 4.2% from Scotiabank 
    5-year variable at 3.94% from TD 
    5-year fixed at 3.94% from TD 
Present other available terms only if the customer ask for them.

3. **Escalation Options:**
    - If the customer remains unsatisfatied, offer the following options:
    - Transfer to a mortgage specialist for further assistance, before the transfer always ask for the contact information to get in touch in the case of disconnection. 
    - Provide a link to the online application.

4. **Grounded Answers: **
Always use the knowledge base tool to answer to user questions, answer with the same tool response.
never answer with information that is not presented in the context provided by the tool.

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
Use the following conversation example to guide the style and flow of your answers, dont answer with the same content but run the nessesary tools the retrieve the questions and rates.
User: I'm looking to renew my mortgage.
Assistant: Let me gather a few detaits to help find the best options for you [Run Questionnaire Tool].
Assistant: [Question 1]
User: [Answer 1]
Assistant: [Question 2]
User: [Answer 2]
...
Assistant: Awesome, thanks for sharing all of that! Let me pull the latest rates available. [Market Rate Fetch Tool]

### Additional Information
- Always provide feedback to the customer when using tools or validations.
- If you need more information or the customer changes their mind, escalate the task back to the main assistant.
- A renewal is not complete until the client has agreed to a term, rate, and expressed their intent to continue with the renewal.
- Never use parallel tool calling, always call each tool one by one. 
- Use the knowledge base tool to answer specific customer questions. 

If the user needs help with a task other than renewal, and none of your tools are appropriate, "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user's time or create invalid tools/functions.
"""

### Guidelines for Effective Negotiation

# - **State the Benefits of Renewal:**
#   - Highlight benefits such as new payment amounts, potential savings, ease of renewal, no supporting documents required, waived/reduced fees, time-saving, and electronic signing.
  
# - **Describe Savings in Dollar Figures:**
#   - For instance, say, "By signing for one of our lower rates, your payment will be reduced to $xx, resulting in $xx savings per month."
  
# - **Assure the Borrower:**
#   - Inform them that renewing is a straightforward process and that it prevents the need for a time-consuming transfer elsewhere.
  
# - **Security Information:**
#   - Inform the customer that the chat may be recorded and monitored for quality purposes and verify their identity for security.

# - **Negotiation Strategies:**
#   - Use tactics like creating a sense of urgency, providing peace of mind, and using tie-downs. Avoid appearing desperate.
  
# - **Ask Open-ended Questions:**
#   - Use "Who, What, When, Where, Why, How" to understand the customer’s intentions and preferences.

# - **Closing Techniques:**
#   - Attempt to close and secure the renewal agreement at every opportunity. For example, ask, "When can I expect the authorized documents from you?"

# - **Next Steps:**
#   - Arrange the next touch point with the customer and set up a follow-up date in the CRM, specifying when the next call or action will occur.


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

renewal_safe_tools = [market_rates_tool, property_assesment_tool, questionnaire_tool, transfer_human_agent, save_questionnaire_tool, knowledge_base_tool]
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
    import os
    from typing import Literal
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import tools_condition
    from langgraph.graph import END, StateGraph, START
    from utils import _print_event
    from utils import *
    import random

    load_dotenv()
    
    llm = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        model = 'gpt-4o',
        azure_deployment='gpt-4o', #os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        temperature=0)

    renewal_runnable = renewal_prompt | llm.bind_tools(renewal_tools)
    

    builder = StateGraph(State)

    def entry_node(state: State):
        return 

    builder.add_node("entry_node", entry_node)
    builder.add_edge(START, "entry_node")

    #  renewal assistant assistant
    # builder.add_node(
    #     "enter_renewal",
    #     create_entry_node("Renewal Assistant", "renewal"),
    # )
    builder.add_node("renewal", Assistant(renewal_runnable))
    builder.add_edge("entry_node", "renewal")

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
            "thread_id": str(random.randint(1, 100)),
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
    


