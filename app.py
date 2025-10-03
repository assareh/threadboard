"""
Threadboard - Generic Reddit Feed Filter Platform

A Flask application that allows users to create custom filtered Reddit feeds.
"""

import os
import json
import time
import logging
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Optional
from threading import Thread

from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('threadboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
FEEDS_DIR = DATA_DIR / "feeds"
POSTS_DIR = DATA_DIR / "posts"
TRACKING_DIR = DATA_DIR / "tracking"

# Create directories
for directory in [DATA_DIR, FEEDS_DIR, POSTS_DIR, TRACKING_DIR]:
    directory.mkdir(exist_ok=True)

# Reddit OAuth configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

# LLM Configuration
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LOCAL_LLM_URL = "http://127.0.0.1:1234/v1/chat/completions"

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Global state for background tasks
feed_tasks = {}


class RedditAPI:
    """Handle Reddit API interactions"""

    def __init__(self):
        self.token = None
        self.token_expires = 0

    def _get_oauth_token(self):
        """Get Reddit OAuth token"""
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            logger.warning("Reddit OAuth not configured, using unauthenticated requests")
            return

        try:
            response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
                data={'grant_type': 'client_credentials'},
                headers={'User-Agent': 'python:threadboard:v1.0.0 (by /u/ThreadboardBot)'},
                timeout=10
            )
            response.raise_for_status()

            token_data = response.json()
            self.token = token_data['access_token']
            self.token_expires = time.time() + token_data.get('expires_in', 3600)
            logger.info("Successfully obtained Reddit OAuth token")

        except Exception as e:
            logger.error(f"Failed to get Reddit OAuth token: {e}")
            self.token = None

    def _ensure_token(self):
        """Ensure we have a valid Reddit OAuth token"""
        if not self.token or time.time() > (self.token_expires - 300):
            self._get_oauth_token()

    @retry(
        retry=retry_if_exception_type(requests.exceptions.HTTPError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def fetch_posts(self, subreddit: str, limit: int = 20) -> List[Dict]:
        """Fetch posts from a subreddit"""
        headers = {'User-Agent': 'python:threadboard:v1.0.0 (by /u/ThreadboardBot)'}

        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            self._ensure_token()
            if self.token:
                headers['Authorization'] = f'Bearer {self.token}'
                url = f'https://oauth.reddit.com/r/{subreddit}/new.json?limit={limit}'
            else:
                url = f'https://www.reddit.com/r/{subreddit}/new.json?limit={limit}'
        else:
            url = f'https://www.reddit.com/r/{subreddit}/new.json?limit={limit}'

        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Reddit rate limit hit, waiting {retry_after}s")
            time.sleep(retry_after)
            response.raise_for_status()

        response.raise_for_status()
        data = response.json()

        if 'data' not in data or 'children' not in data['data']:
            logger.error(f"Unexpected JSON structure from {url}")
            return []

        posts = []
        for child in data['data']['children']:
            post_data = child.get('data', {})
            created_utc = post_data.get("created_utc", 0)

            if created_utc:
                published_dt = datetime.fromtimestamp(float(created_utc), tz=ZoneInfo("America/Los_Angeles"))
                published_str = published_dt.strftime('%B %d, %Y at %I:%M %p PT')
            else:
                published_str = "Unknown"

            post = {
                "id": post_data.get("name", ""),
                "title": post_data.get("title", ""),
                "link": f"https://reddit.com{post_data.get('permalink', '')}",
                "author": post_data.get("author", ""),
                "published": published_str,
                "body": post_data.get("selftext", ""),
                "subreddit": post_data.get("subreddit", ""),
                "subreddit_prefixed": post_data.get("subreddit_name_prefixed", ""),
                "flair": post_data.get('link_flair_text', '') or ''
            }
            posts.append(post)

        logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
        return posts


class LLMFilter:
    """Handle LLM filtering of posts"""

    def query(self, title: str, body: str, filter_prompt: str) -> Optional[Dict]:
        """Query LLM to determine if post is interesting"""
        system_message = f"""You are a content filter that evaluates Reddit posts based on user-defined criteria.

User's filtering criteria:
{filter_prompt}

Evaluate the post and return ONLY valid JSON with this structure:
{{"interesting": true/false, "reason": "brief explanation"}}"""

        user_message = f"""Post Title: {title}

Post Body:
{body}"""

        try:
            if USE_GEMINI and GEMINI_API_KEY:
                return self._query_gemini(system_message, user_message)
            else:
                return self._query_local(system_message, user_message)
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return None

    def _query_gemini(self, system_message: str, user_message: str) -> Optional[Dict]:
        """Query Gemini API"""
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        prompt = f"{system_message}\n\n{user_message}"
        response = model.generate_content(prompt)
        result_text = response.text.strip()

        # Clean up response
        if result_text.startswith("```json"):
            result_text = result_text.split("\n", 1)[1]
            result_text = result_text.rsplit("```", 1)[0].strip()
        elif result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1]
            result_text = result_text.rsplit("```", 1)[0].strip()

        return json.loads(result_text)

    def _query_local(self, system_message: str, user_message: str) -> Optional[Dict]:
        """Query local LLM"""
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "post_filter",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "interesting": {"type": "boolean"},
                            "reason": {"type": "string"}
                        },
                        "required": ["interesting", "reason"],
                        "additionalProperties": False
                    }
                }
            }
        }

        response = requests.post(LOCAL_LLM_URL, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"].strip()

        # Clean up thinking tags if present
        import re
        assistant_message = re.sub(r'<think>.*?</think>', '', assistant_message, flags=re.DOTALL).strip()

        return json.loads(assistant_message)


# Initialize services
reddit_api = RedditAPI()
llm_filter = LLMFilter()


@app.route('/')
def index():
    """Home page showing all feeds"""
    feeds = []
    for feed_file in FEEDS_DIR.glob("*.json"):
        with open(feed_file) as f:
            feed = json.load(f)
            feeds.append(feed)

    feeds.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return render_template('index.html', feeds=feeds)


@app.route('/create', methods=['GET', 'POST'])
def create_feed():
    """Create a new feed"""
    if request.method == 'POST':
        feed_name = request.form['feed_name']
        subreddits = [s.strip() for s in request.form['subreddits'].split(',')]
        frequency = int(request.form['frequency'])
        filter_prompt = request.form['filter_prompt']

        # Generate unique feed ID
        feed_id = str(uuid.uuid4())

        feed_data = {
            'id': feed_id,
            'name': feed_name,
            'subreddits': subreddits,
            'frequency_minutes': frequency,
            'filter_prompt': filter_prompt,
            'created_at': datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
            'last_check': None
        }

        # Save feed configuration
        feed_file = FEEDS_DIR / f"{feed_id}.json"
        with open(feed_file, 'w') as f:
            json.dump(feed_data, f, indent=2)

        # Start background polling task
        start_feed_task(feed_id, frequency)

        logger.info(f"Created new feed: {feed_name} (ID: {feed_id})")
        return redirect(url_for('view_feed', feed_id=feed_id))

    return render_template('create_feed.html')


@app.route('/feed/<feed_id>')
def view_feed(feed_id: str):
    """View a specific feed"""
    # Load feed configuration
    feed_file = FEEDS_DIR / f"{feed_id}.json"
    if not feed_file.exists():
        return "Feed not found", 404

    with open(feed_file) as f:
        feed = json.load(f)

    # Load filtered posts
    posts_file = POSTS_DIR / f"{feed_id}.json"
    posts = []
    if posts_file.exists():
        with open(posts_file) as f:
            posts = json.load(f)

    pacific_time = datetime.now(ZoneInfo("America/Los_Angeles"))

    return render_template(
        'feed.html',
        feed=feed,
        posts=posts,
        now=pacific_time.strftime('%B %d, %Y at %I:%M %p PT')
    )


def start_feed_task(feed_id: str, frequency_minutes: int):
    """Start a background task for a feed"""
    def run_feed_loop():
        while True:
            try:
                process_feed(feed_id)
            except Exception as e:
                logger.error(f"Error processing feed {feed_id}: {e}")

            time.sleep(frequency_minutes * 60)

    if feed_id not in feed_tasks:
        thread = Thread(target=run_feed_loop, daemon=True)
        thread.start()
        feed_tasks[feed_id] = thread
        logger.info(f"Started background task for feed {feed_id} (every {frequency_minutes} minutes)")


def process_feed(feed_id: str):
    """Process a single feed - fetch and filter posts"""
    # Load feed configuration
    feed_file = FEEDS_DIR / f"{feed_id}.json"
    with open(feed_file) as f:
        feed = json.load(f)

    # Load tracking data
    tracking_file = TRACKING_DIR / f"{feed_id}.json"
    tracked_posts = {}
    if tracking_file.exists():
        with open(tracking_file) as f:
            tracked_posts = json.load(f)

    # Load existing filtered posts
    posts_file = POSTS_DIR / f"{feed_id}.json"
    filtered_posts = []
    if posts_file.exists():
        with open(posts_file) as f:
            filtered_posts = json.load(f)

    # Fetch and filter posts from each subreddit
    new_posts_count = 0
    for subreddit in feed['subreddits']:
        posts = reddit_api.fetch_posts(subreddit)

        for post in posts:
            post_id = post['id']

            # Skip if already processed
            if post_id in tracked_posts:
                continue

            # Filter with LLM
            result = llm_filter.query(post['title'], post['body'], feed['filter_prompt'])

            if result and result.get('interesting', False):
                logger.info(f"✨ Interesting post found in r/{subreddit}: {post['title'][:50]}")

                post['reason'] = result.get('reason', 'N/A')
                post['timestamp'] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
                filtered_posts.append(post)
                new_posts_count += 1

            # Mark as tracked
            tracked_posts[post_id] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()

    # Save filtered posts
    with open(posts_file, 'w') as f:
        json.dump(filtered_posts, f, indent=2)

    # Save tracking data
    with open(tracking_file, 'w') as f:
        json.dump(tracked_posts, f, indent=2)

    # Update feed last_check time
    feed['last_check'] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
    with open(feed_file, 'w') as f:
        json.dump(feed, f, indent=2)

    logger.info(f"Processed feed {feed['name']}: {new_posts_count} new posts found")


if __name__ == '__main__':
    # Start background tasks for all existing feeds
    for feed_file in FEEDS_DIR.glob("*.json"):
        with open(feed_file) as f:
            feed = json.load(f)
            start_feed_task(feed['id'], feed['frequency_minutes'])

    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
