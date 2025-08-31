"""Configuration file for Twitter Market Analysis Pipeline."""

import random
from datetime import datetime, timezone

# Scraping Configuration
SEARCH_HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty"]
TARGET_TWEETS = 500  # Minimum tweets to collect
TIME_WINDOW_HOURS = 24  # Last 24 hours
MAX_CONCURRENT_REQUESTS = 2  # Conservative for API limits

# Rate Limiting
MIN_DELAY = 2.0  # Minimum delay between requests (seconds)
MAX_DELAY = 5.0  # Maximum delay between requests (seconds)
REQUEST_TIMEOUT = 30  # Request timeout in seconds

# Data Storage
DATA_DIR = "data"
RAW_DATA_FILE = "raw_tweets.parquet"
SIGNALS_DATA_FILE = "processed_signals.parquet"
OUTPUT_JSON_FILE = "tweets.json"  # For compatibility with original scraper

# TF-IDF Configuration
TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 2  # Minimum document frequency
TFIDF_MAX_DF = 0.8  # Maximum document frequency
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# Signal Generation
SIGNAL_THRESHOLD_POSITIVE = 0.3  # Threshold for BUY signals
SIGNAL_THRESHOLD_NEGATIVE = -0.3  # Threshold for SELL signals
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for trading signals

# Visualization Settings
PLOT_FIGSIZE = (12, 8)
PLOT_DPI = 100
MAX_WORDS_WORDCLOUD = 200

# Custom stop words for Indian stock market
STOP_WORDS_CUSTOM = {
    'rt', 'via', 'amp', 'quot', 'gt', 'lt', 'https', 'http', 'www',
    'com', 'co', 'in', 'the', 'and', 'or', 'but', 'for', 'with',
    'twitter', 'tweet', 'follow', 'retweet', 'share', 'like',
    'today', 'now', 'will', 'get', 'can', 'may', 'should', 'would',
    'one', 'two', 'first', 'second', 'new', 'old', 'good', 'bad',
    'big', 'small', 'high', 'low', 'more', 'less', 'most', 'much',
    'many', 'some', 'all', 'any', 'no', 'not', 'only', 'just',
    'time', 'day', 'week', 'month', 'year', 'hour', 'minute'
}

# Logging Configuration
LOG_LEVEL = "INFO"

# Market Hours (IST)
MARKET_OPEN_HOUR = 9   # 9:15 AM IST
MARKET_CLOSE_HOUR = 15  # 3:30 PM IST

# Tweet Quality Filters
MIN_CONTENT_LENGTH = 10
MAX_CONTENT_LENGTH = 500
MIN_ENGAGEMENT_FOR_ANALYSIS = 1

# Memory Management
MAX_MEMORY_USAGE_MB = 1024  # 1GB limit for processing
CHUNK_SIZE_FOR_PROCESSING = 1000  # Process data in chunks

# Feature Engineering
SENTIMENT_WINDOW_HOURS = 4  # Rolling window for sentiment calculation
TREND_ANALYSIS_PERIODS = [1, 3, 6, 12, 24]  # Hours for trend analysis

# File Paths
COOKIES_FILE = "cookies.json"
LOG_FILE = "scraper.log"
ERROR_LOG_FILE = "errors.log"

# API Configuration (for twikit)
CLIENT_LANGUAGE = 'en-US'
SEARCH_PRODUCT = "Latest"  # Latest tweets
DEFAULT_SEARCH_COUNT = 100  # Per request limit

# Performance Monitoring
ENABLE_PERFORMANCE_LOGGING = True
MEMORY_CHECK_INTERVAL = 100  # Check memory every N tweets processed

# Data Validation
REQUIRED_TWEET_FIELDS = ['username', 'content', 'timestamp', 'id']
REQUIRED_NUMERIC_FIELDS = ['likes', 'retweets', 'replies']

# Export Settings
EXPORT_FORMATS = ['json', 'parquet', 'csv']  # Supported export formats
INCLUDE_DEBUG_DATA = True  # Include debug fields in exports

# Indian Market Specific
INDIAN_MARKET_KEYWORDS = [
    'nifty', 'sensex', 'bse', 'nse', 'rupee', 'inr', 'indian',
    'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata',
    'reliance', 'tcs', 'infosys', 'hdfc', 'icici', 'sbi',
    'adani', 'bajaj', 'tata', 'mahindra', 'wipro', 'ongc'
]

# Signal Strength Categories
SIGNAL_CATEGORIES = {
    'VERY_STRONG': 0.7,
    'STRONG': 0.5,
    'MODERATE': 0.3,
    'WEAK': 0.1,
    'VERY_WEAK': 0.0
}

def get_current_market_session() -> str:
    """Determine current market session based on IST time."""
    # Convert UTC to IST (UTC + 5:30)
    ist_now = datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
    ist_hour = (ist_now.hour + 5) % 24  # Approximate IST conversion
    
    if MARKET_OPEN_HOUR <= ist_hour < MARKET_CLOSE_HOUR:
        return "MARKET_HOURS"
    elif ist_hour < MARKET_OPEN_HOUR:
        return "PRE_MARKET"
    else:
        return "AFTER_MARKET"

def get_dynamic_delay() -> float:
    """Get dynamic delay based on current time and market session."""
    base_delay = random.uniform(MIN_DELAY, MAX_DELAY)
    
    session = get_current_market_session()
    
    # Adjust delay based on market session
    if session == "MARKET_HOURS":
        return base_delay * 0.8  # Faster during market hours
    elif session == "PRE_MARKET":
        return base_delay * 1.2  # Slower pre-market
    else:
        return base_delay  # Normal after hours