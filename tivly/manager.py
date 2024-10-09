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
from dotenv import load_dotenv

try:
    from utils import create_tool_node_with_fallback, CompleteOrEscalate, create_entry_node, State, pop_dialog_state
    from questionaree import *
except ImportError:
    from .utils import create_tool_node_with_fallback, CompleteOrEscalate, create_entry_node, State, pop_dialog_state
    from .questionaree import *

load_dotenv()

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

# Primary Assistant

# The top-level assistant performs general Q&A and delegates specialized tasks to other assistants.
# The task delegation is a simple form of semantic routing / does simple intent detection

llm = AzureChatOpenAI(
    model='gpt-4o',
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment='gpt-4o', #os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=0)

# servicing_runnable = servicing_prompt | llm.bind_tools(
#     servicing_tools + [CompleteOrEscalate]
# )

questionaree_runnable = questionaree_prompt | llm.bind_tools(
    questionaree_tools +  [CompleteOrEscalate]
)

primary_prompt = """ You are a customer service agent for Tivly, a company specializing in connecting small to medium-sized businesses with suitable insurance providers.

Guidelines:
1. Greet the customer warmly and mention Tivly's core service.
2. Offer help and ask how you can help the customer.
3. Provide information about the types of insurance products offered, such as general liability, workers' compensation, commercial property insurance, and professional liability.
4. If the customer is asking for a new policy for a bussiness, ask if he would like to proceed with the telephone interview and run the questionaree tool if the customer agrees.

Note: Tivly caters to a diverse range of industries, including 18 Wheeler Insurance, Beauty Professionals, Carpenters, Chiropractors, Cleaning Companies, Construction &amp; Contractors, Consultants, Estheticians, Microblading, Online Businesses, Personal Trainers, Photography, Restaurants, Retail, Tree Trimming, Trucking, Vending Machine Operators, and Warehouses.

Example Interaction:
1. **Agent**: "Hello! Welcome to Tivly, your trusted partner in finding the perfect insurance for your business."
2. **Customer**: "Hi, I'm a personal trainer looking for professional liability insurance."
3. **Agent**: "Great to hear from you! We offer a wide range of insurance products including professional liability insurance specifically for personal trainers. Can you tell me a bit more about your business needs?"
4. **Customer**: "Sure, I work with clients both in-person and online."
5. **Agent**: "That's fantastic. Our professional liability insurance for personal trainers can cover both in-person and online sessions. Let's connect you with the best providers for your needs."

If a customer requests help to adquire an insurance for his bussiness, delegate the task to the appropriate specialized assistant by invoking the corresponding tool. 
You are not able to make these types of process by yourself. Only the specialized assistants are given permission to do this for the user.
The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. 
Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. 
\nCurrent time: {time} """# ".",

# Listening: Practice active listening, not just hearing. Take notes and pay full attention to the borrower. Avoid interruptions and paraphrase their statements to ensure understanding.
# Pursuing: Incorporate an elevator speech in your conversation. Actively pursue questionarees and close all gaps. Create a sense of urgency using FOMO, FUD, benefit statements, and quantifying the benefits. Always ask to get what you need and confirm the next steps, setting a follow-up date with the borrower.


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
    # search_user_info,
    # rag_tool,
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        # ToServicingAssistant,
        ToQuestionareeAssistant
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

# def user_info(state: State):
#     return {"user_info": search_user_info.invoke({})}

# builder.add_node("fetch_user_info", user_info)


# questionaree assistant 
builder.add_node("enter_questionaree", create_entry_node("Questionaree Assistant", "questionaree"))
builder.add_node("questionaree", Assistant(questionaree_runnable))
builder.add_edge("enter_questionaree", "questionaree")
builder.add_node("questionaree_sensitive_tools", create_tool_node_with_fallback(questionaree_sensitive_tools))
builder.add_node("questionaree_safe_tools", create_tool_node_with_fallback(questionaree_safe_tools))
builder.add_edge("questionaree_sensitive_tools", "questionaree")
builder.add_edge("questionaree_safe_tools", "questionaree")
builder.add_conditional_edges("questionaree", route_questionaree)

builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")


# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node( "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))

def route_primary_assistant(
    state: State,
) -> Literal[
    "primary_assistant_tools",
    "enter_questionaree",
    # "enter_book_excursion",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        # if tool_calls[0]["name"] == ToServicingAssistant.__name__:
        #     return "enter_servicing"
        if tool_calls[0]["name"] == ToQuestionareeAssistant.__name__:
            return "enter_questionaree"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")

# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        # "enter_servicing": "enter_servicing",
        "enter_questionaree": "enter_questionaree",
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
    # "servicing",
    "questionaree",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]

# builder.add_conditional_edges("fetch_user_info", route_to_workflow)
builder.add_conditional_edges(START, route_to_workflow)

# Compile graph
memory = MemorySaver()
agent = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        # "servicing_sensitive_tools",
        "questionaree_sensitive_tools",
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
        "user_id": "3442 587242",
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