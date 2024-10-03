import streamlit as st
from openai import OpenAI
from mortgage_agent.utils import _print_ai_message, _print_event
# from tivly.manager import agent
from typing import List, Generator
import numpy as np
from schema import ChatMessage
from PIL import Image
# from st_audiorec import st_audiorec
from audiorecorder import audiorecorder
from stt import transcribe_audio_from_memory, transcribe_audio_from_file
from tts import text_to_speech
import base64
from time import sleep
from pydub import AudioSegment
import io
import os 

os.environ["AZURE_OPENAI_API_VERSION"] = st.secrets["AZURE_OPENAI_API_VERSION"]
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
os.environ["AZURE_SPEECH_KEY"] = st.secrets["AZURE_SPEECH_KEY"]
os.environ["AZURE_REGION"] = st.secrets["AZURE_REGION"]
os.environ['POSTGRES_REMOTE_ENDPOINT'] = st.secrets['POSTGRES_REMOTE_ENDPOINT']
os.environ['POSTGRES_REMOTE_USER'] = st.secrets['POSTGRES_REMOTE_USER']
os.environ['POSTGRES_REMOTE_PASSWORD'] = st.secrets['POSTGRES_REMOTE_PASSWORD']
os.environ['POSTGRES_DB_NAME'] = st.secrets['POSTGRES_DB_NAME']
os.environ['POSTGRES_SSL_MODE'] = st.secrets['POSTGRES_SSL_MODE']

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()

        # Step 2: Load the audio data into a BytesIO object
        audio_stream = io.BytesIO(data)
        # Step 3: Load the audio into pydub's AudioSegment
        audio = AudioSegment.from_file(audio_stream)
        # Step 4: Get the duration in milliseconds and convert to seconds
        duration_seconds = len(audio) / 1000
        print(f"Duration: {duration_seconds} seconds")

        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )
    return duration_seconds

def draw_messages(
    messages_iter,
    is_new=False,
    chat=None
):
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    # streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    # while msg := await anext(messages_agen, None):
    while msg := next(messages_iter, None):
        # str message represents an intermediate token being streamed
        # if isinstance(msg, str):
        #     # If placeholder is empty, this is the first token of a new message
        #     # being streamed. We need to do setup.
        #     if not streaming_placeholder:
        #         if last_message_type != "ai":
        #             last_message_type = "ai"
        #             st.session_state.last_message = st.chat_message("ai")
        #         with st.session_state.last_message:
        #             streaming_placeholder = st.empty()

        #     streaming_content += msg
        #     streaming_placeholder.write(streaming_content)
        #     continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                # pass
                last_message_type = "human"
                chat.chat_message("human", avatar=human_avatar).write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = chat.chat_message("ai", avatar=ai_avatar)


                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                        if is_new: #if is a new message, generate the voice a play it
                            text_to_speech(msg.content)
                            duration = autoplay_audio("output.wav")
                            if duration >= 3:
                                sleep(duration)
                            else:
                                sleep(duration-3)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = next(messages_iter)
                            if not tool_result.type == "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


def message_generator(messages) -> Generator[ChatMessage, None, None]:
    for mes in messages:
        yield ChatMessage.from_langchain(mes)

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "customer_id": 1,
        # Checkpoints are accessed by thread_id
        "thread_id": '124',
    }
}
# _printed = set()
_loggs = set()
st.title("BL Agents")

# with st.sidebar:
#     wav_audio_data = st_audiorec()

#     if wav_audio_data is not None:
#         st.audio(wav_audio_data, format='audio/wav')


human_avatar = Image.open("./human_asset.png")
ai_avatar = Image.open("./ai_asset.png")

# option = st.selectbox(
#     "Mortgage Agent",
#     ("Mortgage Agent", "Tivly Agent"),
# )

# if option == 'Mortgage Agent':
from mortgage_agent.assistant import agent
selected_agent = agent 
# elif option == 'Tivly Agent': 
#     from tivly.manager import agent
#     selected_agent = agent 

def reset_state():
    st.session_state.messages =  []
    st.session_state._printed = set()

# Initialize chat history
if "messages" not in st.session_state:
    reset_state()
messages: List[ChatMessage] = st.session_state.messages

chat = st.container(height=400)

# st.button('Reset', on_click=reset_state)

if len(messages) == 0:
    WELCOME = "Welcome to BLBank. How can I assist you today?"
    with chat.chat_message("ai", avatar=ai_avatar):
        st.write(WELCOME)

print("Initialized state:", st.session_state.messages)

def amessage_iter(messages):
        for m in messages:
            yield m

draw_messages(amessage_iter(messages), chat=chat)


# Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

    # React to user input
    

# msg, rec = st.columns([0.5, 0.5])

# with msg:
msg_input = st.chat_input("How can I help you?")
# with rec:
audio = audiorecorder(start_prompt="", stop_prompt="", pause_prompt="", show_visualizer=True, key=None)

# elif transcription:

transcription = None
if len(audio) > 0: 
    print(audio)
    audio.export("audio.wav", format="wav")
    # st.audio(audio.export().read()) 
    transcription = transcribe_audio_from_file("audio.wav")
    print(transcription)
    
    # cshat.chat_message("human", avatar=human_avatar).write(transcription)
    # user_input = transcription

user_input = None
if msg_input:
    user_input = msg_input
elif transcription:
    user_input = transcription

if user_input :#:
    # Display user message in chat message container
    # st.chat_message("user").markdown(user_input)
    # # Add user message to chat history
    # st.session_state.messages.append({"role": "user", "content": user_input})

    messages.append(ChatMessage(type="human", content=user_input))
    chat.chat_message("human", avatar=human_avatar).write(user_input)

    # print(f"Echo: {user_input}")

    # Display assistant response in chat message container
    # with st.chat_message("assistant"):
    events = selected_agent.stream(
            {"messages": ("user", user_input)}, config, stream_mode="values"
        )
 
    for event in events:
        pass
        # _print_event(event, _loggs) #logs event
        # last_messages = event.get('messages')
    # print(f'last messages {event}')
  
    
    new_events = _print_ai_message(event, st.session_state._printed) #message to end user
    # print(f"_printed : {st.session_state._printed}")
    print(f'new events to draw: {new_events}')
    draw_messages(message_generator(new_events[1:]), is_new=True, chat=chat)


