import os
import time
import speech_recognition as sr
import wave
import numpy as np
import pyaudio
import tempfile
import argparse
import logging
import random
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Configure logging to write to a file
logging.basicConfig(filename='call_summarizer.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize OpenAI client with base URL support for LiteLLM
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

DATA_DIR = "DATA"
os.makedirs(DATA_DIR, exist_ok=True)

class CallSummarizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.p = pyaudio.PyAudio()  # Store instance globally
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
        
        stream = self.p.open(format=self.audio_format,
                             channels=self.channels,
                             rate=self.sample_rate,
                             input=True,
                             frames_per_buffer=self.chunk_size)
        
        logging.info("Recording started. Press Ctrl+C to stop recording...")
        print("Recording started. Press Ctrl+C to stop recording...")

        start_time = time.time()
        
        try:
            while self.recording:
                data = stream.read(self.chunk_size)
                self.audio_data.append(data)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(int(elapsed_time), 60)
                print(f"\rRecording... {minutes}m {seconds}s elapsed. Press Ctrl+C to stop.", end="")
                
                time.sleep(1)  # Update every second
        except KeyboardInterrupt:
            logging.info("Recording stopped.")
            print("\nRecording stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            self.stop_recording()
            
    def stop_recording(self):
        """Stop recording audio"""
        self.recording = False
        self.p.terminate()  # Ensures it's always closed
        
    def save_audio(self, filename=None):
        """Save recorded audio to a WAV file"""
        if not self.audio_data:
            logging.warning("No audio data to save")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(DATA_DIR, f"call_recording_{timestamp}.wav")
            
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.audio_data))
        wf.close()
        
        logging.info(f"Audio saved to {filename}")
        return filename
    
    def validate_audio_file(self, file_path):
        """Validate that the WAV file is correctly formatted"""
        try:
            with wave.open(file_path, "rb") as wf:
                # Check if the file has the correct number of channels and sample width
                if wf.getnchannels() != self.channels or wf.getsampwidth() != pyaudio.PyAudio().get_sample_size(self.audio_format):
                    logging.error(f"Invalid WAV file format: {file_path}")
                    return False
                logging.info(f"Valid WAV file: {file_path}")
                return True
        except Exception as e:
            logging.exception(f"Error validating WAV file: {file_path}")
            return False
    
    def check_llm_availability(self):
        """Check the availability of the configured LLM"""
        try:
            # Make a simple API call to check availability
            client.models.list()
            logging.info("LLM is available.")
            return True
        except Exception as e:
            logging.exception("Error checking LLM availability")
            return False
    
    def transcribe_audio(self, audio_file, chunk_duration=10, total_duration=None, keep_temp_files=False):
        """Transcribe audio file to text using OpenAI's Whisper API"""
        if not self.validate_audio_file(audio_file):
            logging.error("Transcription aborted due to audio file issues.")
            return None
        
        def clean_transcript(transcript):
            """Cleans up Whisper transcription by removing filler words and noise markers."""
            transcript = transcript.replace("[inaudible]", "").replace("[background noise]", "")
            transcript = transcript.replace("uh", "").replace("um", "")  # Remove fillers (optional)
            return transcript.strip()
        
        
        try:
            with wave.open(audio_file, "rb") as wf:
                total_frames = wf.getnframes()
                frame_rate = wf.getframerate()
                chunk_size = frame_rate * chunk_duration
                if total_duration:
                    total_frames = min(total_frames, frame_rate * total_duration)
                transcript = []

                for start in tqdm(range(0, total_frames, chunk_size), desc="Chunking"):
                    wf.setpos(start)
                    frames = wf.readframes(chunk_size)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                        temp_wav_name = temp_wav.name
                        # Write frames to a valid WAV file format
                        with wave.open(temp_wav_name, "wb") as temp_wav_file:
                            temp_wav_file.setnchannels(self.channels)
                            temp_wav_file.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
                            temp_wav_file.setframerate(self.sample_rate)  # Enforce correct rate
                            temp_wav_file.writeframes(frames)

                    # Validate the temporary WAV file
                    if not self.validate_audio_file(temp_wav_name):
                        logging.warning("Skipping invalid chunk.")
                        continue

                    retries = 3
                    for attempt in range(retries):
                        try:
                            with open(temp_wav_name, "rb") as audio:
                                chunk_transcript = client.audio.transcriptions.create(
                                    model=os.getenv("TRANSCRIPTION_MODEL", "whisper-1"),
                                    file=audio
                                )
                            if chunk_transcript.text:
                                transcript.append(chunk_transcript.text)
                            break
                        except Exception as e:
                            logging.exception("Error transcribing chunk")
                            if attempt < retries - 1:
                                sleep_time = min(5, (2 ** attempt) + random.uniform(0, 1))
                                logging.info(f"Retrying in {sleep_time:.2f} seconds...")
                                time.sleep(sleep_time)
                    if not keep_temp_files:
                        os.remove(temp_wav_name)

            final_transcript = clean_transcript(" ".join(transcript))
            return final_transcript
        except Exception as e:
            logging.exception("Error processing audio file")
            return None
    
    def summarize_text(self, text):
        """Generate a summary of the transcribed text using OpenAI API"""
        if not text:
            return "No text to summarize"
        
        verbosity = os.getenv("SUMMARY_VERBOSITY", "detailed")  # Default to detailed
        if verbosity == "short":
            system_prompt = "Summarize phone calls into 3 bullet points. Focus only on decisions and action items."
        else:
            system_prompt = "Summarize phone calls into key points using bullet points. Highlight action items, decisions, and important topics. Keep it clear and concise."
        
        retries = 3
        for attempt in range(retries):
            try:
                model = os.getenv("SUMMARY_MODEL", "gpt-3.5-turbo")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Summarize this phone call:\n\n{text}\n\n### Summary:"}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.exception("Error generating summary")
                if attempt < retries - 1:
                    sleep_time = min(5, (2 ** attempt) + random.uniform(0, 1))
                    logging.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
        return "Failed to generate summary."
    
    def summarize_large_text(self, text, max_chunk_size=3000):
        """Breaks long transcripts into smaller parts, summarizes each, and then summarizes all summaries."""
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        summaries = []
        
        for chunk in chunks:
            messages = [
                {"role": "system", "content": "Summarize this conversation clearly, highlighting key points, decisions, and action items."},
                {"role": "user", "content": f"Summarize this part:\n\n{chunk}\n\n### Summary:"}
            ]
            response = client.chat.completions.create(model="gpt-4o", messages=messages)
            summaries.append(response.choices[0].message.content)

        # Final summary of all summaries
        final_summary = self.summarize_text(" ".join(summaries))
        return final_summary
    
    def process_call(self, audio_file=None, verbose=False, chunk_duration=10, total_duration=None, keep_temp_files=False):
        """Record, transcribe, and summarize a call"""
        try:
            if not self.check_llm_availability():
                logging.error("LLM is not available. Please check your configuration.")
                return
            
            if audio_file:
                logging.info(f"Analyzing provided audio file: {audio_file}")
            else:
                try:
                    self.start_recording()
                except KeyboardInterrupt:
                    logging.info("Recording interrupted.")
                    return
                audio_file = self.save_audio()
                if not audio_file:
                    return
            logging.info("Transcribing audio...")
            transcript = self.transcribe_audio(audio_file, chunk_duration, total_duration, keep_temp_files)
            if not transcript:
                logging.error("Transcription failed.")
                return
            
            # Print transcript only if verbose flag is set
            if verbose:
                logging.info("\nTranscript:\n")
                logging.info(transcript)
            
            # Save transcript to a Markdown file with the prefix "Transcript"
            transcript_md_filename = os.path.join(DATA_DIR, os.path.splitext(os.path.basename(audio_file))[0] + "_Transcript.md")
            with open(transcript_md_filename, "w") as md_file:
                md_file.write("# Call Transcript\n\n")
                md_file.write(transcript)
            logging.info(f"Transcript saved to {transcript_md_filename}")
            print(f"Transcript saved to {transcript_md_filename}")
            
            logging.info("\nGenerating hierarchical summary...\n")
            summary = self.summarize_large_text(transcript)
            
            logging.info("\nCall Summary:\n")
            logging.info(summary)
            
            # Save summary to a Markdown file with the prefix "Summary"
            summary_md_filename = os.path.join(DATA_DIR, os.path.splitext(os.path.basename(audio_file))[0] + "_Summary.md")
            with open(summary_md_filename, "w") as md_file:
                md_file.write("# Call Summary\n\n")
                md_file.write(summary)
            logging.info(f"Summary saved to {summary_md_filename}")
            print(f"Summary saved to {summary_md_filename}")
            
            print(f"Files created:\nTranscript: {transcript_md_filename}\nSummary: {summary_md_filename}")
            
            return {
                "audio_file": audio_file,
                "transcript": transcript,
                "summary": summary
            }
        except KeyboardInterrupt:
            logging.info("Process interrupted.")
            return

def display_help():
    """Display help information for the CLI"""
    help_text = """
    Usage: python call_summarizer.py [options] <path_to_wav_file>

    Options:
      --help                  Show this help message and exit
      --verbose               Print transcript to console
      --chunk_duration=<sec>  Specify chunk duration for transcription (default: 10 seconds)
      --total_duration=<sec>  Specify total duration of call snippet to transcribe
      --debug                 Keep temporary WAV files during development

    Examples:
      python call_summarizer.py ./DATA/call1.wav --verbose
      python call_summarizer.py ./DATA/call1.wav --chunk_duration=60 --total_duration=300 --debug

    Note: The summarization process now uses hierarchical summarization to handle long transcripts effectively.
    """
    logging.info(help_text)

def main():
    logging.info("Call Summarizer - Record and summarize your calls")
    
    summarizer = CallSummarizer()
    parser = argparse.ArgumentParser(description='Call Summarizer - Record and summarize your calls')
    parser.add_argument('audio_file', nargs='?', help='Path to the WAV file')
    parser.add_argument('--verbose', action='store_true', help='Print transcript to console')
    parser.add_argument('--chunk_duration', type=int, help='Specify chunk duration for transcription')
    parser.add_argument('--total_duration', type=int, help='Specify total duration of call snippet to transcribe')
    parser.add_argument('--debug', action='store_true', help='Keep temporary WAV files during development')
    parser.set_defaults(chunk_duration=10)  # Default value for chunk_duration
    args = parser.parse_args()

    summarizer.process_call(args.audio_file, args.verbose, args.chunk_duration, args.total_duration, args.debug)

if __name__ == "__main__":
    main()
