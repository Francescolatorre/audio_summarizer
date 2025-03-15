import pyttsx3
import wave

# Sample text to be spoken
text = """This is a test recording. The purpose of this file is to verify transcription accuracy.
We are testing different words, phrases, and sentence structures. Let's check how well the transcription works."""

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking rate

# Save spoken text as WAV file
audio_filename = "DATA/test_audio.wav"
engine.save_to_file(text, audio_filename)
engine.runAndWait()

print(f"WAV file '{audio_filename}' generated successfully with the transcript below:\n")
print(text)
