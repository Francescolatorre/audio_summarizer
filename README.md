# Call Summarizer

An application that records audio from your microphone, transcribes it, and generates a summary of your calls.

## Features

- Record audio from your microphone
- Transcribe audio using OpenAI's Whisper API
- Generate concise summaries of call content using OpenAI's GPT model
- Save recordings for future reference
- Validate audio files for correct format
- Hierarchical summarization for long transcripts

## Requirements

- Python 3.7+
- OpenAI API key
- Microphone connected to your computer

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/callsummarizer.git
   cd callsummarizer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your configuration:
   ```
   OPENAI_API_KEY=your_api_key_here
   OPENAI_API_BASE=http://your-litellm-server:8000/v1
   SUMMARY_MODEL=gpt-3.5-turbo
   TRANSCRIPTION_MODEL=whisper-1
   ```

   If using the official OpenAI API, you can omit the `OPENAI_API_BASE` variable.

## Usage

Run the application:
```
python call_summarizer.py
```

- The program will start recording from your microphone
- Press Ctrl+C to stop recording
- The application will transcribe the audio and generate a summary
- Both the transcript and summary will be displayed in the console
- The audio recording is saved to a file with a timestamp
- Transcripts and summaries are saved as Markdown files

## Notes

- This application requires a working microphone and internet connection
- The quality of transcription and summary depends on audio quality
- OpenAI API usage will incur costs based on your OpenAI account
- Ensure the `.env` file is correctly configured for API access

## License

[MIT License](LICENSE)
