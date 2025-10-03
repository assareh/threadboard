# Threadboard

**Generic Reddit Feed Filter Platform** - Create personalized, AI-filtered Reddit feeds based on your interests.

Threadboard is a Flask web application that monitors multiple subreddits and uses Large Language Models (LLMs) to intelligently filter posts based on custom criteria you define. Instead of scrolling through hundreds of posts, let AI curate the content that matters to you.

## Features

- **Custom Feed Creation**: Build unlimited personalized feeds from any combination of subreddits
- **AI-Powered Filtering**: Use natural language to describe what you're interested in
- **Flexible LLM Support**: Choose between Google Gemini or local LLM (via LM Studio)
- **Automatic Monitoring**: Background tasks continuously check for new posts
- **Clean Web Interface**: Modern, responsive UI with dark theme support
- **Persistent Storage**: All feeds and filtered posts saved locally
- **Reddit OAuth Support**: Optional OAuth for higher rate limits

## Use Cases

- **Job Hunting**: Monitor job boards like r/forhire, r/hiring for specific roles and skills
- **Deal Finding**: Track r/buildapcsales, r/deals for specific products or price ranges
- **Learning**: Aggregate educational content from multiple programming subreddits
- **Market Research**: Follow industry trends across multiple niche communities
- **Content Discovery**: Find specific types of content (tutorials, reviews, discussions)

## Quick Start (Local)

### Prerequisites

- Python 3.9+
- (Optional) [LM Studio](https://lmstudio.ai/) for local LLM, or Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd threadboard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the web interface**
   ```
   Open http://localhost:5000 in your browser
   ```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Required
SECRET_KEY=your-secret-key-here

# Optional - Reddit OAuth (recommended for higher rate limits)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# LLM Configuration
USE_GEMINI=false  # Set to "true" for Gemini, "false" for local LLM
GEMINI_API_KEY=your_api_key  # Required if USE_GEMINI=true
```

### Reddit OAuth Setup (Optional)

Higher rate limits are available with Reddit OAuth:

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in the form:
   - **name**: Threadboard
   - **redirect uri**: http://localhost:8080 (not used but required)
5. Copy the client ID (under the app name) and secret
6. Add to your `.env` file

### LLM Configuration

#### Option 1: Google Gemini (Recommended for beginners)

1. Get an API key from https://makersuite.google.com/app/apikey
2. Set in `.env`:
   ```
   USE_GEMINI=true
   GEMINI_API_KEY=your_key_here
   ```

#### Option 2: Local LLM (Privacy-focused, free)

1. Install [LM Studio](https://lmstudio.ai/)
2. Download a model (e.g., Qwen 2.5 Coder 7B, Llama 3.2)
3. Start the local server (default: http://127.0.0.1:1234)
4. Set in `.env`:
   ```
   USE_GEMINI=false
   ```

## Feed Creation Flow

1. **Navigate to "Create New Feed"**
2. **Fill in the form**:
   - **Feed Name**: Descriptive name (e.g., "Python Jobs")
   - **Subreddits**: Comma-separated list (e.g., "python,django,flask")
   - **Check Frequency**: How often to check (5-1440 minutes)
   - **Filter Criteria**: Natural language description of what you want

3. **Example Filter Criteria**:
   ```
   Remote software engineering positions requiring Python,
   focusing on backend or data engineering roles. Prefer
   senior-level positions with salary above $100k.
   ```

4. **Submit** - Your feed starts monitoring immediately!

5. **View Results** - Click on your feed to see filtered posts

## How It Works

### Architecture

```
┌─────────────────┐
│   Flask Web UI  │  ← User creates feeds and views results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feed Manager   │  ← Manages feed configurations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Background Tasks│  ← Polls Reddit on schedule
└────────┬────────┘
         │
         ├────────▶ ┌──────────────┐
         │          │  Reddit API  │  ← Fetches new posts
         │          └──────────────┘
         │
         └────────▶ ┌──────────────┐
                    │  LLM Filter  │  ← Evaluates posts
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Filtered     │
                    │ Posts Storage│
                    └──────────────┘
```

### Process Flow

1. **Feed Creation**: User defines subreddits and filtering criteria
2. **Background Polling**: Daemon threads check each feed on schedule
3. **Post Fetching**: Retrieve latest posts from Reddit API
4. **Duplicate Tracking**: Skip posts that were already evaluated
5. **LLM Filtering**: Each new post is evaluated against filter criteria
6. **Storage**: Matching posts saved to JSON files
7. **Display**: Web UI shows filtered posts with AI reasoning

### Data Storage

All data is stored locally in the `data/` directory:

```
data/
├── feeds/          # Feed configurations (JSON)
├── posts/          # Filtered posts per feed (JSON)
└── tracking/       # Processed post IDs to avoid duplicates (JSON)
```

## Development

### Project Structure

```
threadboard/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── .gitignore            # Git ignore rules
├── templates/            # HTML templates
│   ├── base.html        # Base template with theme
│   ├── index.html       # Feed list
│   ├── create_feed.html # Feed creation form
│   └── feed.html        # Individual feed view
└── data/                # Runtime data (gitignored)
    ├── feeds/
    ├── posts/
    └── tracking/
```

### Adding Features

The codebase is designed to be extensible:

- **New LLM Providers**: Add methods to `LLMFilter` class
- **Reddit Enhancement**: Extend `RedditAPI` class
- **UI Improvements**: Edit templates (all use CSS variables for theming)
- **Feed Management**: Add routes in `app.py`

### Running Tests

```bash
# Run with debug mode
python app.py

# Check logs
tail -f threadboard.log
```

## API Rate Limits

- **Without OAuth**: ~60 requests/hour per subreddit
- **With OAuth**: ~600 requests/hour per subreddit
- **Recommendation**: Use OAuth if monitoring 5+ subreddits or checking frequently

## Troubleshooting

### "No posts found yet"

- Check that background tasks are running (check logs)
- Verify subreddit names are correct (no r/ prefix needed)
- Wait for first polling cycle to complete
- Check Reddit API status

### LLM Connection Issues

**Local LLM:**
- Ensure LM Studio server is running
- Verify URL in `.env` matches LM Studio settings
- Check model is loaded and ready

**Gemini:**
- Verify API key is correct
- Check API quota/limits
- Ensure `USE_GEMINI=true` in `.env`

### High Resource Usage

- Increase check frequency (reduce polling frequency)
- Monitor fewer subreddits
- Use smaller local LLM models

## Security Notes

- Never commit `.env` file
- Keep `SECRET_KEY` secure in production
- Reddit OAuth credentials are sensitive
- LLM API keys should be protected

## Contributing

Contributions welcome! Areas for improvement:

- Additional LLM provider support (Anthropic Claude, OpenAI, etc.)
- Feed sharing/export functionality
- Email/webhook notifications for new posts
- Advanced filtering (regex, keyword exclusions)
- Performance optimizations
- Docker deployment support

## License

MIT License - See LICENSE file for details

## Credits

Built with:
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Reddit API](https://www.reddit.com/dev/api/) - Content source
- [Google Gemini](https://ai.google.dev/) - AI filtering (optional)
- [LM Studio](https://lmstudio.ai/) - Local LLM support (optional)

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review logs in `threadboard.log`
