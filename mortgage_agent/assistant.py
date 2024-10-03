from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
# from langchain_anthropic import ChatAnthropic
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig
import datetime 
from langchain_openai import AzureChatOpenAI
import os
# from dotenv import load_dotenv

try:
    # Try relative import when used as part of a package
    from .utils import create_tool_node_with_fallback, CompleteOrEscalate, create_entry_node, State, pop_dialog_state
    from .servicing import *
    from .renewal import *
    from .policies import rag_tool
    from .tools import search_user_info
    
except ImportError:
    # Fallback to absolute import when running directly
    from utils import create_tool_node_with_fallback, CompleteOrEscalate, create_entry_node, State, pop_dialog_state
    from servicing import *
    from renewal import *
    from policies import rag_tool
    from tools import search_user_info

# load_dotenv()

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    
    # async def __call__(self, state: State, config: RunnableConfig): 
    #     response = await self.runnable.ainvoke(state, config)

    #     if state["is_last_step"] and response.tool_calls:
    #         return {
    #             "messages": [
    #                 AIMessage(
    #                     id=response.id,
    #                     content="Sorry, need more steps to process this request.",
    #                 )
    #             ]
    #         }
    #     return {"messages": [response]}

# Primary Assistant

# The top-level assistant performs general Q&A and delegates specialized tasks to other assistants.
# The task delegation is a simple form of semantic routing / does simple intent detection

llm = AzureChatOpenAI(
    model='gpt-4o',
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment='gpt-4o', #os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=0)

servicing_runnable = servicing_prompt | llm.bind_tools(
    servicing_tools + [CompleteOrEscalate]
)

renewal_runnable = renewal_prompt | llm.bind_tools(
    renewal_tools +  [CompleteOrEscalate]
)

## Before we proceed, I need to verify your identity for security purposes. Please provide the necessary information as per our validation policy. 
primary_prompt = """ You are a customer support assistant at BLBank whose primary role is to help clients with their banking needs and to answer questions about the company's policies.

Use the following guidelines to provide the best customer experience:

1. Greeting: Greet the customer appropriately Good [morning/afternoon] and offer help.
2. Security: Please note, this chat may be recorded and monitored for accuracy, service quality, and training purposes. Thank you.
3. Presence: Show enthusiasm and interest in your interactions. Maintain a positive tone and steady pace. Remember to use verbal manners and always smile.
4. Relating: Listen to the borrower's needs and try to understand their perspective. Show empathy and acknowledge their concerns.
5. Questioning: Ask open-ended questions to gather more information. Probe and clarify to ensure you understand their situation and can provide the best solution. Keep your language simple and clear.

Special Instructions:
- If the customer requests to perform a renewal or a servicing-related task such as lump sum payment or title change, delegate the task to the appropriate specialized assistant by invoking the corresponding tool.
- You are not able to process these types of requests yourself. Only the specialized assistants have the permission to handle these tasks.
- Do not mention the transfer to the specialized assistants to the customer; just quietly delegate through function calls.
- Provide detailed information to the customer and always double-check the database before concluding that any information is unavailable.

Response style:
Adopt a more conversational-speech style, suitable for integration with Speech-to-Text (STT) and Text-to-Speech (TTS) systems.
To achieve this, please keep the following guidelines in mind:

- Use a friendly and approachable tone, similar to natural spoken conversation.
- Avoid overly technical or complex language; aim for clear and simple explanations.
- Use contractions (e.g., “don’t” instead of “do not”) to mirror natural speech.
- Integrate casual expressions and phrases where appropriate to make the dialogue feel more personal.
- Keep responses concise and to the point, but ensure they remain informative and helpful.
- Please respond to customer inquiries while adhering to these guidelines.

Begin by greeting the customer and wait for their response. Start every response with a greeting and follow the guidelines provided.
\n\nCurrent user information:\n\n{user_info}\n
\nCurrent time: {time} """# ".",

# Listening: Practice active listening, not just hearing. Take notes and pay full attention to the borrower. Avoid interruptions and paraphrase their statements to ensure understanding.
# Pursuing: Incorporate an elevator speech in your conversation. Actively pursue renewals and close all gaps. Create a sense of urgency using FOMO, FUD, benefit statements, and quantifying the benefits. Always ask to get what you need and confirm the next steps, setting a follow-up date with the borrower.


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            primary_prompt
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.datetime.now())

primary_assistant_tools = [
    # TavilySearchResults(max_results=1),
    search_user_info,
    rag_tool,
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        AssignToServicingAssistant,
        AssignToRenewalAssistant
    ]
)

### Utility 

from typing import Callable
from langchain_core.messages import ToolMessage
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)

def user_info(state: State):
    return {"user_info": search_user_info.invoke({})}

builder.add_node("fetch_user_info", user_info)

# builder.set_entry_point("fetch_user_info")
builder.add_edge(START, "fetch_user_info")


#  servicing assistant assistant
builder.add_node("enter_servicing", create_entry_node("Servicing Assistant", "servicing"))
builder.add_node("servicing", Assistant(servicing_runnable))
builder.add_edge("enter_servicing", "servicing")
builder.add_node("servicing_sensitive_tools",create_tool_node_with_fallback(servicing_sensitive_tools))
builder.add_node( "servicing_safe_tools", create_tool_node_with_fallback(servicing_safe_tools))
builder.add_edge("servicing_sensitive_tools", "servicing")
builder.add_edge("servicing_safe_tools", "servicing")
builder.add_conditional_edges("servicing", route_servicing)

builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")

# renewal assistant 
builder.add_node("enter_renewal", create_entry_node("Renewals Assistant", "renewal"))
builder.add_node("renewal", Assistant(renewal_runnable))
builder.add_edge("enter_renewal", "renewal")
builder.add_node("renewal_sensitive_tools", create_tool_node_with_fallback(renewal_sensitive_tools))
builder.add_node("renewal_safe_tools", create_tool_node_with_fallback(renewal_safe_tools))
builder.add_edge("renewal_sensitive_tools", "renewal")
builder.add_edge("renewal_safe_tools", "renewal")
builder.add_conditional_edges("renewal", route_renewal)


# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node( "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))

def route_primary_assistant(
    state: State,
) -> Literal[
    "primary_assistant_tools",
    "enter_servicing",
    "enter_renewal",
    # "enter_book_excursion",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == AssignToServicingAssistant.__name__:
            return "enter_servicing"
        elif tool_calls[0]["name"] == AssignToRenewalAssistant.__name__:
            return "enter_renewal"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")

# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        "enter_servicing": "enter_servicing",
        "enter_renewal": "enter_renewal",
        "primary_assistant_tools": "primary_assistant_tools",
        END: END,
    },
)
builder.add_edge("primary_assistant_tools", "primary_assistant")

# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "servicing",
    "renewal",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]

builder.add_conditional_edges("fetch_user_info", route_to_workflow)

# Compile graph
memory = MemorySaver()
agent = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        "servicing_sensitive_tools",
        "renewal_sensitive_tools",
        # "book_hotel_sensitive_tools",
        # "book_excursion_sensitive_tools",
    ],
)

if __name__== "__main__" :
    from utils import _print_event
    _printed = set()

    config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "customer_id": 1,
        # Checkpoints are accessed by thread_id
        "thread_id": '124',
    }
}
    
    events = agent.stream(
        {"messages": ("user", "Hi")}, config, stream_mode="values"
        )

    for event in events:
        _print_event(event, _printed)

    while True:

        

        question = input('Input: ')

        events = agent.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)

        snapshot = agent.get_state(config)
        while snapshot.next:
            # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
            # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
            # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
            if user_input.strip() == "y":
                # Just continue
                result = agent.invoke(
                    None,
                    config,
                )
            else:
                # Satisfy the tool invocation by
                # providing instructions on the requested changes / change of mind
                result = agent.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
            snapshot = agent.get_state(config)





# if __name__ == "__main__":
#     import asyncio
#     from uuid import uuid4
#     from dotenv import load_dotenv

#     load_dotenv()

#     config = {
#     "configurable": {
#         # The passenger_id is used in our flight tools to
#         # fetch the user's flight information
#         "customer_id": 1,
#         # Checkpoints are accessed by thread_id
#         "thread_id": '124',
#     }
# }

#     async def main():
#         inputs = {"messages": [("user", "Hi I want to renew my mortgage")]}
#         result = await agent.ainvoke(
#             inputs,
#             config=RunnableConfig(configurable=config),
#         )
#         print('runned')
#         result["messages"][-1].pretty_print()

#         # Draw the agent graph as png
#         # requires:
#         # brew install graphviz
#         # export CFLAGS="-I $(brew --prefix graphviz)/include"
#         # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
#         # pip install pygraphviz
#         #
#         # research_assistant.get_graph().draw_png("agent_diagram.png")

#     asyncio.run(main())
