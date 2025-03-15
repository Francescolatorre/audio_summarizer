import os
import sys
import subprocess
import pytest
import re
import pyttsx3
from difflib import SequenceMatcher

DATA_DIR = "DATA"
AUDIO_FILE = os.path.join(DATA_DIR, "test_audio.wav")
EXPECTED_TRANSCRIPT = """In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep.
And the Spirit of God moved upon the face of the waters. And God said, Let there be light: and there was light. And God saw the light, that it was good:
and God divided the light from the darkness. And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day."""

SIMILARITY_THRESHOLD = 0.85  # Acceptable similarity (85%)

def generate_test_audio():
    """Generate a WAV file with spoken text for transcription testing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust speaking rate
    engine.save_to_file(EXPECTED_TRANSCRIPT, AUDIO_FILE)
    engine.runAndWait()

    assert os.path.exists(AUDIO_FILE), "❌ Error: Audio file was not created."
    print(f"✅ WAV file '{AUDIO_FILE}' generated successfully.")

def get_similarity(a, b):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def normalize_text(text):
    """Convert text to lowercase and remove punctuation for better similarity comparison."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def run_transcription_test():
    """Run the call summarizer script and validate transcription accuracy."""
    assert os.path.exists(AUDIO_FILE), f"❌ Error: '{AUDIO_FILE}' does not exist. Generate audio first."
    
    command = f"python call_summarizer.py {AUDIO_FILE} --verbose"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    assert result.returncode == 0, "❌ Error: Call summarizer script failed."
    
    print("\n📝 Transcription Output:")
    print(result.stdout)
    
    # Extract transcribed text from output
    transcribed_text = result.stdout.strip()
    
    assert transcribed_text, "❌ Error: No transcription output received."
    
    # Calculate similarity
    normalized_transcript = normalize_text(transcribed_text)
    normalized_expected = normalize_text(EXPECTED_TRANSCRIPT)
    similarity = get_similarity(normalized_expected, normalized_transcript)
    print(f"\n🔍 Similarity Score: {similarity:.2f}")
    
    assert similarity >= SIMILARITY_THRESHOLD, f"❌ Error: Transcription accuracy is too low ({similarity:.2f}). Expected ≥ {SIMILARITY_THRESHOLD}."
    
    print("✅ Test Passed: Transcription accuracy meets the threshold.")
def test_transcription_accuracy():
    """Pytest-compatible test for transcription accuracy."""
    assert os.path.exists(AUDIO_FILE), f"❌ Error: '{AUDIO_FILE}' does not exist. Generate audio first."
    
    command = f"python call_summarizer.py {AUDIO_FILE} --verbose"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    assert result.returncode == 0, "❌ Error: Call summarizer script failed."
    
    transcribed_text = result.stdout.strip()
    assert transcribed_text, "❌ Error: No transcription output received."
    
    normalized_transcript = normalize_text(transcribed_text)
    normalized_expected = normalize_text(EXPECTED_TRANSCRIPT)
    similarity = get_similarity(normalized_expected, normalized_transcript)
    assert similarity >= SIMILARITY_THRESHOLD, f"❌ Error: Transcription accuracy is too low ({similarity:.2f}). Expected ≥ {SIMILARITY_THRESHOLD}."

if __name__ == "__main__":
    generate_test_audio()
    run_transcription_test()

def test_missing_audio_file():
    """Test handling of missing audio file."""
    missing_audio_file = os.path.join(DATA_DIR, "non_existent.wav")
    command = f"python call_summarizer.py {missing_audio_file} --verbose"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode != 0, "❌ Error: Expected failure due to missing audio file."

def test_short_audio_transcription():
    """Test transcription of very short audio."""
    short_audio_file = os.path.join(DATA_DIR, "short_test_audio.wav")
    short_transcript = "Hello."
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.save_to_file(short_transcript, short_audio_file)
    engine.runAndWait()
    
    command = f"python call_summarizer.py {short_audio_file} --verbose"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, "❌ Error: Call summarizer script failed for short audio."
    transcribed_text = result.stdout.strip()
    assert transcribed_text, "❌ Error: No transcription output received for short audio."
