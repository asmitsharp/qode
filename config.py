

import os
from datetime import datetime, timedelta

TWITTER_BASE_URL = "https://x.com"
SEARCH_HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty"]
TARGET_TWEETS = 100
TIME_WINDOW_HOURS = 24

MIN_DELAY = 2
MAX_DELAY = 5
RETRY_ATTEMPTS = 3
BACKOFF_FACTOR = 2
REQUEST_TIMEOUT = 30
MAX_CONCURRENT_REQUESTS = 3

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0"
]

DATA_DIR = "data"
RAW_DATA_FILE = "raw_tweets.parquet"
PROCESSED_DATA_FILE = "processed_tweets.parquet"
SIGNALS_DATA_FILE = "trading_signals.parquet"

STOP_WORDS_CUSTOM = {
    'rt', 'amp', 'via', 'http', 'https', 'www', 'com', 'co', 'in',
    'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at',
    'from', 'by', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', 'should', 'now'
}

# TF-IDF configuration
TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8
TFIDF_NGRAM_RANGE = (1, 2)

SIGNAL_THRESHOLD_POSITIVE = 0.6
SIGNAL_THRESHOLD_NEGATIVE = -0.6
CONFIDENCE_INTERVALS = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ

PLOT_DPI = 100
PLOT_FIGSIZE = (12, 8)
MAX_WORDS_WORDCLOUD = 100

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
