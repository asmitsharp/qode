import re
import hashlib
import logging
import asyncio
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
import pandas as pd
try:
    from fake_useragent import UserAgent
    HAS_FAKE_USERAGENT = True
except ImportError:
    HAS_FAKE_USERAGENT = False

def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("scraper.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    
    return text

def extract_hashtags(text: str) -> List[str]:
    return re.findall(r'#(\w+)', text, re.IGNORECASE)

def extract_mentions(text: str) -> List[str]:
    return re.findall(r'@(\w+)', text)

def generate_tweet_hash(content: str, username: str, timestamp: str) -> str:
    combined = f"{username}_{timestamp}_{content}"
    return hashlib.md5(combined.encode()).hexdigest()

def parse_engagement_metrics(text: str) -> Dict[str, int]:
    metrics = {'likes': 0, 'retweets': 0, 'replies': 0}
    
    like_match = re.search(r'(\d+(?:,\d+)*)\s*(?:likes?|hearts?)', text, re.IGNORECASE)
    if like_match:
        metrics['likes'] = int(like_match.group(1).replace(',', ''))
    
    retweet_match = re.search(r'(\d+(?:,\d+)*)\s*(?:retweets?|rts?)', text, re.IGNORECASE)
    if retweet_match:
        metrics['retweets'] = int(retweet_match.group(1).replace(',', ''))
    
    reply_match = re.search(r'(\d+(?:,\d+)*)\s*(?:replies?|comments?)', text, re.IGNORECASE)
    if reply_match:
        metrics['replies'] = int(reply_match.group(1).replace(',', ''))
    
    return metrics

def validate_tweet_data(tweet: Dict[str, Any]) -> bool:
    required_fields = ['username', 'content', 'timestamp']
    return all(field in tweet and tweet[field] for field in required_fields)

def deduplicate_tweets(tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_hashes = set()
    unique_tweets = []
    
    for tweet in tweets:
        tweet_hash = generate_tweet_hash(
            tweet.get('content', ''),
            tweet.get('username', ''),
            str(tweet.get('timestamp', ''))
        )
        
        if tweet_hash not in seen_hashes:
            seen_hashes.add(tweet_hash)
            tweet['hash'] = tweet_hash
            unique_tweets.append(tweet)
    
    return unique_tweets

def create_search_url(hashtag: str, since_date: str = None) -> str:
    base_url = "https://x.com/search"
    query = f"{hashtag} -filter:replies"
    
    if since_date:
        query += f" since:{since_date}"
    
    return f"{base_url}?q={quote_plus(query)}&src=typed_query&f=live"

async def async_sleep_with_jitter(min_delay: float, max_delay: float):
    delay = random.uniform(min_delay, max_delay)
    await asyncio.sleep(delay)

def get_random_user_agent() -> str:
    if HAS_FAKE_USERAGENT:
        try:
            ua = UserAgent()
            return ua.random
        except Exception:
            pass
    
    # Fallback to predefined user agents
    import random
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
    ]
    return random.choice(user_agents)

def safe_extract_text(element, default: str = "") -> str:
    try:
        if element:
            return element.get_text(strip=True)
        return default
    except Exception:
        return default

def parse_relative_time(time_str: str) -> Optional[datetime]:
    if not time_str:
        return None
    
    now = datetime.now(timezone.utc)
    
    time_patterns = {
        r'(\d+)s': lambda x: now - pd.Timedelta(seconds=int(x)),
        r'(\d+)m': lambda x: now - pd.Timedelta(minutes=int(x)),
        r'(\d+)h': lambda x: now - pd.Timedelta(hours=int(x)),
        r'(\d+)d': lambda x: now - pd.Timedelta(days=int(x)),
    }
    
    for pattern, calc_func in time_patterns.items():
        match = re.search(pattern, time_str.lower())
        if match:
            return calc_func(match.group(1))
    
    return None

def calculate_tweet_age_hours(timestamp: datetime) -> float:
    if not timestamp:
        return 0.0
    
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    delta = now - timestamp
    return delta.total_seconds() / 3600

def market_sentiment_keywords():
    return {
        'bullish': ['bull', 'bullish', 'moon', 'rocket', 'up', 'rise', 'high', 'gain', 'profit', 'buy', 'long', 'green', 'surge'],
        'bearish': ['bear', 'bearish', 'crash', 'down', 'fall', 'low', 'loss', 'sell', 'short', 'red', 'dump', 'drop'],
        'neutral': ['sideways', 'flat', 'range', 'consolidation', 'wait', 'watch', 'hold']
    }

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]