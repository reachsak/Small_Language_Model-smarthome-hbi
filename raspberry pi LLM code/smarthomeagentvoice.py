from llama_cpp import Llama
from typing import Union
import requests
from web3 import Web3
from yeelight import Bulb
import pyaudio
import wave
import speech_recognition as sr
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import create_dynamic_model_from_function

# Initialize Web3 instance
web3 = Web3(Web3.HTTPProvider("https://sepolia.infura.io/v3/29ed729d7617400ab427b75b16c31c63"))

def turn_on_light(inner_thoughts: str, command: str) -> str:
    """
    Control the Yeelight bulb.

    Parameters:
        inner_thoughts (str): Inner thoughts to return alongside the action.
        command (str): The command to execute, either "turn on" or "turn off".
        
    Returns:
        str: A message indicating the action taken.
    """
    bulb_ip = "192.168.31.171"  # IP address of the Yeelight bulb
    bulb = Bulb(bulb_ip)
    if command == "turn off":
        bulb.turn_off()
        return f"{inner_thoughts} Light turned off."
    elif command == "turn on":
        bulb.turn_on()
        return f"{inner_thoughts} Light turned on."
    else:
        return "Invalid command. Please provide 'turn on' or 'turn off'."

def turn_off_light(inner_thoughts: str, command: str) -> str:
    """
    Control the Yeelight bulb.

    Parameters:
        inner_thoughts (str): Inner thoughts to return alongside the action.
        command (str): The command to execute, either "turn on" or "turn off".
        
    Returns:
        str: A message indicating the action taken.
    """
    bulb_ip = "192.168.31.171"  # IP address of the Yeelight bulb
    bulb = Bulb(bulb_ip)
    if command == "turn off":
        bulb.turn_off()
        return f"{inner_thoughts} Light turned off."
    elif command == "turn on":
        bulb.turn_on()
        return f"{inner_thoughts} Light turned on."
    else:
        return "Invalid command. Please provide 'turn on' or 'turn off'."

def turn_off_light(inner_thoughts: str, command: str) -> str:
    """
    Control the Yeelight bulb.

    Parameters:
        inner_thoughts (str): Inner thoughts to return alongside the action.
        command (str): The command to execute, either "turn on" or "turn off".
        
    Returns:
        str: A message indicating the action taken.
    """
    bulb_ip = "192.168.31.171"  # IP address of the Yeelight bulb
    bulb = Bulb(bulb_ip)
    if command == "turn off":
        bulb.turn_off()
        return f"{inner_thoughts} Light turned off."
    elif command == "turn on":
        bulb.turn_on()
        return f"{inner_thoughts} Light turned on."
    else:
        return "Invalid command. Please provide 'turn on' or 'turn off'."

DynamicSampleModel3 = create_dynamic_model_from_function(turn_off_light, "turn off")
DynamicSampleModel4 = create_dynamic_model_from_function(turn_on_light, "turn on")
function_tools = [LlamaCppFunctionTool(DynamicSampleModel3), LlamaCppFunctionTool(DynamicSampleModel4)]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)

system_prompt = "You are a smart home AI assistant, you will assist me with controlling the smart home device.\n"

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                                system_prompt=system_prompt + function_tool_registry.get_documentation(),
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)

def record_audio(filename, duration, channels=1, rate=44100, chunk=1024):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
    print("Recording...")
    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

while True:
    # Record audio from the microphone
    filename = "user_input.wav"
    duration = 5  # seconds
    record_audio(filename, duration)
    
    # Transcribe the recorded audio
    user_input1 = transcribe_audio(filename)
    print(f"User: {user_input1}")
    
    if user_input1.lower() == "exit":
        break
    
    user_input = llama_cpp_agent.get_chat_response(user_input1, temperature=0.45, function_tool_registry=function_tool_registry)
    print("AI: " + str(user_input[0]['return_value']))