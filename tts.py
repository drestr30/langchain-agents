import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch Azure Speech key and region from .env file
subscription_key = os.getenv("AZURE_SPEECH_KEY")
region = os.getenv("AZURE_REGION")

def text_to_speech(text, output_audio_file="output.wav"):
    """
    Convert the given text to speech and save the audio to a file.

    :param text: The input text string to be converted to speech.
    :param output_audio_file: The path where the output audio will be saved (default: "output.wav").
    """
    # Initialize the speech configuration with your subscription key and region
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)

    # Set the output format to be an audio file (WAV)
    audio_output = speechsdk.AudioConfig(filename=output_audio_file)

    # Create a synthesizer object for speech synthesis
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)

    # Convert the text to speech
    result = speech_synthesizer.speak_text_async(text).get()

    # Handle the result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized and saved to {output_audio_file}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}. Error details: {cancellation_details.error_details}")

def synthesize_voice_to_memory(text):
    """
    Convert the given text to speech and return the audio data in memory.

    :param text: The input text string to be converted to speech.
    :return: A bytes object representing the synthesized audio.
    """
    # Initialize the speech configuration with your subscription key and region
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)

    # Create an in-memory stream to hold the audio data
    stream = speechsdk.AudioDataStream()

    # Create a synthesizer object for speech synthesis
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    # Convert the text to speech
    result = synthesizer.speak_text_async(text).get()

    # Check the result and store audio in memory
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data_stream = speechsdk.AudioDataStream(result)
        audio_data_stream.save_to_wav_stream(stream)  # Save the audio data to the in-memory stream
        return stream.read_data()  # Return the audio bytes
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        raise RuntimeError(f"Speech synthesis canceled: {cancellation_details.reason}. Error details: {cancellation_details.error_details}")

# Example usage:
# text = "Hello, this is a sample text to speech conversion using Azure."

# # Save the synthesized speech to a WAV file
# text_to_speech(text, output_audio_file="output_audio.wav")

# Get the synthesized speech as in-memory audio (bytes)
# audio_data = synthesize_voice_to_memory(text)
# print(f"Audio synthesized with {len(audio_data)} bytes of data.")
