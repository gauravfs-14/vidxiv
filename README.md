# VidXiv - ArXiv Paper to Video Generator

🎥 Convert ArXiv research papers into engaging video presentations automatically!

![VidXiv Streamlit UI](/static/streamlit_ui.png)
*Fig. VidXiv Streamlit UI*

## Features

- 📄 Fetches papers directly from ArXiv using paper ID
- 🤖 Uses AI (Gemini) to generate video scripts from paper content
- 🎬 Creates multi-scene videos with text overlays
- 🔊 Generates narration using text-to-speech
- 📱 Supports both landscape (YouTube) and portrait (Shorts/Reels) formats
- 🎵 Optional background music support

## Upcoming Features

🚀 **Coming Soon:**

- ✏️ **Script Editing Interface** - Review and modify AI-generated scripts before video creation
- 🖼️ **Automatic Figure Integration** - Smart extraction and placement of paper figures, charts, and diagrams
- 🎨 **Manual Graphics Upload** - Add custom images, logos, and visual elements to enhance presentations
- 🎭 **Multiple AI Voices** - Choose from different TTS voices and speaking styles
- 📊 **Advanced Templates** - Pre-designed video templates for different research fields
- 🔄 **Batch Processing** - Generate videos for multiple papers simultaneously
- 🌐 **Multi-language Support** - Generate videos in different languages

## Installation

```bash
uv sync
```

## Setup

1. Copy the environment template:

```bash
cp .env.template .env
```

2. Edit `.env` and add your API keys if needed (for Gemini or other LLM models)

## Usage

1. Start the Streamlit app:

```bash
streamlit run main.py
```

2. Open your browser to the displayed URL (usually `http://localhost:8501`)

3. Enter an ArXiv paper ID (e.g., `2401.06015`)

4. Choose video format:
   - Uncheck for landscape YouTube format (16:9)
   - Check for portrait Shorts/Reels format (9:16)

5. Optionally upload background music (MP3 format)

6. Click "Generate Video" and wait for processing

7. Download your generated video!

## Requirements

- Python 3.11+
- Internet connection (for fetching papers and AI processing)
- Sufficient disk space for temporary video files

## Dependencies

- `arxiv` - Fetching papers from ArXiv
- `pymupdf` - PDF processing and figure extraction
- `gtts` - Text-to-speech for narration
- `moviepy` - Video editing and composition
- `streamlit` - Web interface
- `langchain` - LLM integration
- `requests` - HTTP requests
- `pillow` - Image processing
- `python-dotenv` - Environment variable management

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed correctly
2. **MoviePy errors**: Try installing with `pip install moviepy[optional]`
3. **Font errors**: Install system fonts or use the fallback font options
4. **Memory issues**: Try with shorter papers or reduce video quality

### Error Messages

- "Could not add background music": The background music file may be corrupted or in an unsupported format
- "Error generating video": Check that all dependencies are properly installed and try again

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - see LICENSE file for details
