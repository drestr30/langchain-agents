import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv
import io

# Load environment variables from .env file
load_dotenv()

# Fetch Azure Speech key and region from .env file
subscription_key = os.getenv("AZURE_SPEECH_KEY")
region = os.getenv("AZURE_REGION")

def transcribe_audio_from_file(audio_file_path):
    # Initialize the speech configuration with your subscription key and region
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    
    # Set up the audio configuration using the file path
    audio_input = speechsdk.AudioConfig(filename=audio_file_path)

    # Create a speech recognizer
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    # Start speech recognition and get the result
    result = recognizer.recognize_once()

    # Handle result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech could be recognized."
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        return f"Speech Recognition canceled: {cancellation_details.reason}. Error details: {cancellation_details.error_details}"

def transcribe_audio_from_memory(audio_data):
    # Initialize the speech configuration with your subscription key and region
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)

    # Set up the audio configuration using in-memory audio data
    audio_input = speechsdk.AudioConfig(stream=speechsdk.AudioInputStream(stream=audio_data))

    # Create a speech recognizer
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    # Start speech recognition and get the result
    result = recognizer.recognize_once()

    # Handle result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech could be recognized."
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        return f"Speech Recognition canceled: {cancellation_details.reason}. Error details: {cancellation_details.error_details}"

# Example usage:
# # Transcribing audio from a file
# audio_file_path = "path_to_your_audio_file.wav"  # Replace with your local audio file path
# transcription = transcribe_audio_from_file(audio_file_path)
# print("Transcription from file:", transcription)

# # Transcribing in-memory audio
# # Assuming you have loaded your audio file into memory as bytes
# with open("path_to_your_audio_file.wav", "rb") as audio_file:
#     audio_bytes = io.BytesIO(audio_file.read())

# transcription_memory = transcribe_audio_from_memory(audio_bytes)
# print("Transcription from memory:", transcription_memory)
