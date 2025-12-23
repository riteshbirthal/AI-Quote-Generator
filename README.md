# AI Quote Reel Bot

Automatically generate motivational quote reels for YouTube Shorts using Google's Gemini AI.

## Features

- **AI Quote Generation** - Original quotes using Gemini 2.5/3.0
- **Multi-language Support** - English and Hindi quotes
- **Epic Background Music** - Auto-generated cinematic music based on quote mood
- **Text-to-Speech** - High-quality voiceover using Edge-TTS
- **Auto Model Switching** - Automatically switches Gemini models when quota exhausted
- **YouTube Metadata** - Generates titles, descriptions, tags, and keywords
- **Batch Processing** - Create multiple reels at once

## Requirements

- Python 3.10+
- FFmpeg (for audio mixing)
- Google AI API Key ([Get one here](https://aistudio.google.com/apikey))

## Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-quote-reel-bot.git
cd ai-quote-reel-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg
# Windows: winget install ffmpeg
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Docker Setup

```bash
# Build image
docker build -t ai-quote-reel-bot .

# Run container
docker run -v $(pwd)/output:/app/output \
    -e GOOGLE_API_KEY=your-api-key \
    ai-quote-reel-bot --mode single --language english
```

## Configuration

Create a `.env` file with:

```env
# Required
GOOGLE_API_KEY=your-google-api-key-here

# Optional
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

## Usage

### Generate Single Reel

```bash
# English reel
python main.py --mode single --language english

# Hindi reel
python main.py --mode single --language hindi

# Specific category
python main.py --mode single --language english --category motivation
```

### Generate Batch

```bash
# Generate 5 English reels
python main.py --mode batch --count 5 --language english

# Generate 10 Hindi reels
python main.py --mode batch --count 10 --language hindi
```

### Available Categories

- motivation
- success
- mindset
- discipline
- perseverance
- wisdom
- leadership
- growth

## Output

### Files

```
output/
├── reels/              # Generated MP4 videos (1080x1920)
├── thumbnails/         # Video thumbnails
└── reels_data.json     # YouTube metadata for all reels
```

### Reel Data JSON

Each reel includes YouTube-ready metadata:

```json
{
  "reel_id": "abc123",
  "language": "english",
  "quote": {
    "text": "Your quote here...",
    "category": "motivation",
    "hook": "Listen to this",
    "hashtags": ["motivation", "success", ...]
  },
  "files": {
    "video": "output/reels/reel_abc123.mp4",
    "thumbnail": "output/thumbnails/reel_abc123_thumb.jpg"
  },
  "youtube": {
    "title": "🔥 Listen to this | Motivation #shorts",
    "description": "Full description with hashtags...",
    "tags": ["motivation", "quotes", ...],
    "keywords": "motivation, success, quotes..."
  }
}
```

## Project Structure

```
ai-quote-reel-bot/
├── main.py              # Main pipeline
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── Dockerfile           # Docker config
├── quote_engine/        # Gemini quote generation
├── audio_engine/        # TTS + background music
├── video_engine/        # Video composition
├── seo_engine/          # YouTube metadata
├── output/              # Generated reels
└── temp/                # Temporary files
```

## Auto Model Switching

When API quota is exhausted, the bot automatically tries these models in order:

1. gemini-2.5-flash
2. gemini-2.0-flash
3. gemini-2.0-flash-lite
4. gemini-2.5-pro
5. gemini-3-flash-preview
6. gemini-3-pro-preview
7. gemini-flash-latest
8. gemini-exp-1206

## API Limits

Free tier limits (per day):
- ~20 requests per model
- Auto-switching helps maximize free usage

For higher limits, upgrade at [Google AI Studio](https://aistudio.google.com/).

## License

MIT License
