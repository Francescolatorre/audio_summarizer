import os
import time
import speech_recognition as sr
import wave
import numpy as np
import pyaudio
import tempfile
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with base URL support for LiteLLM
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

class CallSummarizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.recording = True
        self.audio_data = []
        
        p = pyaudio.PyAudio()
        stream = p.open(format=self.audio_format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)
        
        print("Recording started. Press Ctrl+C to stop recording...")
        
        try:
            while self.recording:
                data = stream.read(self.chunk_size)
                self.audio_data.append(data)
        except KeyboardInterrupt:
            print("Recording stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.recording = False
            
    def stop_recording(self):
        """Stop recording audio"""
        self.recording = False
        
    def save_audio(self, filename=None):
        """Save recorded audio to a WAV file"""
        if not self.audio_data:
            print("No audio data to save")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"call_recording_{timestamp}.wav"
            
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.audio_data))
        wf.close()
        
        print(f"Audio saved to {filename}")
        return filename
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text using OpenAI's Whisper API"""
        try:
            with open(audio_file, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model=os.getenv("TRANSCRIPTION_MODEL", "whisper-1"),
                    file=audio
                )
            return transcript.text
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            print("If using LiteLLM, ensure it supports audio transcription")
            return None
    
    def summarize_text(self, text):
        """Generate a summary of the transcribed text using OpenAI API"""
        if not text:
            return "No text to summarize"
            
        try:
            model = os.getenv("SUMMARY_MODEL", "gpt-3.5-turbo")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes phone calls. Create a concise summary of the key points, action items, and important information from the conversation."},
                    {"role": "user", "content": f"Please summarize the following phone call transcript:\n\n{text}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Failed to generate summary."
    
    def process_call(self):
        """Record, transcribe, and summarize a call"""
        try:
            self.start_recording()
        except KeyboardInterrupt:
            pass
        
        audio_file = self.save_audio()
        if not audio_file:
            return
            
        print("Transcribing audio...")
        transcript = self.transcribe_audio(audio_file)
        if not transcript:
            print("Transcription failed.")
            return
            
        print("\nTranscript:")
        print(transcript)
        
        print("\nGenerating summary...")
        summary = self.summarize_text(transcript)
        
        print("\nCall Summary:")
        print(summary)
        
        return {
            "audio_file": audio_file,
            "transcript": transcript,
            "summary": summary
        }

def main():
    print("Call Summarizer - Record and summarize your calls")
    print("Make sure you have set your OPENAI_API_KEY in a .env file")
    print("Press Ctrl+C to stop recording")
    
    summarizer = CallSummarizer()
    summarizer.process_call()

if __name__ == "__main__":
    main()