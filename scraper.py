"""
Twitter scraper for Indian stock market tweets using twikit.
Integrated with existing pipeline architecture.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import re
import json
from twikit import Client

from config import *
from utils import (
    setup_logging, extract_hashtags, extract_mentions, 
    generate_tweet_hash, validate_tweet_data, deduplicate_tweets,
    calculate_tweet_age_hours
)

class TwitterScraper:
    """Twitter scraper using twikit with anti-bot detection and rate limiting."""
    
    def __init__(self):
        self.logger = setup_logging(LOG_LEVEL)
        self.client = Client(language='en-US')
        self.scraped_tweets = []
        self.session_requests = 0
        self.session_start = time.time()
        self.authenticated = False
        
    async def initialize_client(self) -> bool:
        """Initialize and authenticate twikit client."""
        try:
            # Try to load existing cookies
            self.client.load_cookies("cookies.json")
            self.logger.info("Loaded existing cookies for authentication")
            self.authenticated = True
            return True
            
        except Exception:
            self.logger.info("No existing cookies found, attempting guest login...")
            try:
                await self.client.login_guest()
                self.client.save_cookies("cookies.json")
                self.logger.info("Guest login successful, cookies saved")
                self.authenticated = True
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to authenticate with Twitter: {e}")
                self.authenticated = False
                return False
    
    def extract_tweet_data(self, tweet, hashtag_source: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from twikit Tweet object."""
        try:
            # Get timestamp with fallback handling
            try:
                tweet_time = tweet.created_at_datetime
            except AttributeError:
                try:
                    tweet_time = datetime.fromisoformat(tweet.created_at.replace('Z', '+00:00'))
                except:
                    tweet_time = datetime.now(timezone.utc)
            
            # Extract basic tweet data
            tweet_text = getattr(tweet, 'text', '') or getattr(tweet, 'full_text', '')
            username = getattr(tweet.user, 'screen_name', 'unknown') if hasattr(tweet, 'user') else 'unknown'
            
            # Extract engagement metrics with fallbacks
            likes = getattr(tweet, 'favorite_count', 0) or 0
            retweets = getattr(tweet, 'retweet_count', 0) or 0
            replies = getattr(tweet, 'reply_count', 0) or 0
            
            # Extract mentions and hashtags from text
            mentions = extract_mentions(tweet_text)
            hashtags = extract_hashtags(tweet_text)
            
            # Generate hash for deduplication
            tweet_hash = generate_tweet_hash(tweet_text, username, tweet_time.isoformat())
            
            data = {
                'id': str(tweet.id),
                'hash': tweet_hash,
                'username': username,
                'content': tweet_text,
                'timestamp': tweet_time,
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'total_engagement': likes + retweets + replies,
                'mentions': mentions,
                'hashtags': hashtags,
                'hashtag_source': hashtag_source,
                'url': f"https://twitter.com/{username}/status/{tweet.id}",
                'scraped_at': datetime.now(timezone.utc),
                'content_length': len(tweet_text),
                'hashtag_count': len(hashtags),
                'mention_count': len(mentions)
            }
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Error extracting tweet data: {e}")
            return None
    
    def is_recent_tweet(self, timestamp: datetime) -> bool:
        """Check if tweet is within the target time window."""
        if not timestamp:
            return False
        
        age_hours = calculate_tweet_age_hours(timestamp)
        return age_hours <= TIME_WINDOW_HOURS
    
    async def scrape_hashtag(self, hashtag: str, max_tweets_per_hashtag: int) -> List[Dict[str, Any]]:
        """Scrape tweets for a specific hashtag using twikit."""
        if not self.authenticated:
            if not await self.initialize_client():
                return []
        
        self.logger.info(f"Scraping hashtag: {hashtag} (target: {max_tweets_per_hashtag} tweets)")
        
        tweets = []
        seen_ids = set()
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=TIME_WINDOW_HOURS)
        
        try:
            # Search for tweets
            search_results = await self.client.search_tweet(
                hashtag, 
                product="Latest", 
                count=min(max_tweets_per_hashtag, 100)  # API limit per request
            )
            
            if not search_results:
                self.logger.warning(f"No tweets found for {hashtag}")
                return []
            
            for tweet in search_results:
                # Skip if already processed
                if tweet.id in seen_ids:
                    continue
                
                # Extract tweet data
                tweet_data = self.extract_tweet_data(tweet, hashtag)
                if not tweet_data:
                    continue
                
                # Check if tweet is recent enough
                if not self.is_recent_tweet(tweet_data['timestamp']):
                    self.logger.debug(f"Tweet {tweet.id} is too old, skipping...")
                    continue
                
                # Validate tweet data
                if not validate_tweet_data(tweet_data):
                    continue
                
                tweets.append(tweet_data)
                seen_ids.add(tweet.id)
                
                if len(tweets) >= max_tweets_per_hashtag:
                    break
            
            # Add delay to respect rate limits
            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            
        except Exception as e:
            self.logger.error(f"Error scraping hashtag {hashtag}: {e}")
        
        self.logger.info(f"Collected {len(tweets)} tweets for hashtag: {hashtag}")
        return tweets
    
    async def scrape_hashtag_with_pagination(self, hashtag: str, max_tweets: int) -> List[Dict[str, Any]]:
        """Scrape hashtag with pagination to get more tweets."""
        if not self.authenticated:
            if not await self.initialize_client():
                return []
        
        tweets = []
        seen_ids = set()
        cursor = None
        requests_made = 0
        max_requests = 5  # Limit API requests to avoid rate limiting
        
        self.logger.info(f"Starting paginated scraping for {hashtag}")
        
        while len(tweets) < max_tweets and requests_made < max_requests:
            try:
                if cursor:
                    # Get next page
                    search_results = await cursor.next()
                else:
                    # Initial search
                    search_results = await self.client.search_tweet(
                        hashtag,
                        product="Latest",
                        count=100
                    )
                    cursor = search_results
                
                requests_made += 1
                
                if not search_results:
                    break
                
                page_tweets = 0
                for tweet in search_results:
                    if tweet.id in seen_ids:
                        continue
                    
                    tweet_data = self.extract_tweet_data(tweet, hashtag)
                    if not tweet_data:
                        continue
                    
                    if not self.is_recent_tweet(tweet_data['timestamp']):
                        # If we hit old tweets, we can stop paginating
                        self.logger.info(f"Reached old tweets for {hashtag}, stopping pagination")
                        return tweets
                    
                    if validate_tweet_data(tweet_data):
                        tweets.append(tweet_data)
                        seen_ids.add(tweet.id)
                        page_tweets += 1
                    
                    if len(tweets) >= max_tweets:
                        break
                
                self.logger.info(f"Page {requests_made}: collected {page_tweets} tweets for {hashtag}")
                
                # Rate limiting delay
                await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                
            except Exception as e:
                self.logger.error(f"Error in pagination for {hashtag}: {e}")
                break
        
        return tweets
    
    async def scrape_all_hashtags(self) -> List[Dict[str, Any]]:
        """Scrape tweets from all configured hashtags."""
        if not await self.initialize_client():
            self.logger.error("Failed to initialize Twitter client")
            return []
        
        self.logger.info("Starting comprehensive tweet scraping")
        self.logger.info(f"Target hashtags: {SEARCH_HASHTAGS}")
        self.logger.info(f"Target tweets: {TARGET_TWEETS}")
        self.logger.info(f"Time window: {TIME_WINDOW_HOURS} hours")
        
        all_tweets = []
        tweets_per_hashtag = max(TARGET_TWEETS // len(SEARCH_HASHTAGS), 100)
        
        # Sequential scraping to avoid overwhelming the API
        for hashtag in SEARCH_HASHTAGS:
            try:
                self.logger.info(f"Scraping {hashtag}...")
                
                # Use pagination for better coverage
                hashtag_tweets = await self.scrape_hashtag_with_pagination(
                    hashtag, 
                    tweets_per_hashtag
                )
                
                if hashtag_tweets:
                    all_tweets.extend(hashtag_tweets)
                    self.logger.info(f"Successfully scraped {len(hashtag_tweets)} tweets from {hashtag}")
                else:
                    self.logger.warning(f"No tweets collected for {hashtag}")
                
                # Longer delay between hashtags
                await asyncio.sleep(random.uniform(3, 6))
                
            except Exception as e:
                self.logger.error(f"Failed to scrape hashtag {hashtag}: {e}")
                continue
        
        # Final deduplication across all hashtags
        self.logger.info(f"Deduplicating {len(all_tweets)} collected tweets...")
        all_tweets = deduplicate_tweets(all_tweets)
        
        # Additional validation
        valid_tweets = [tweet for tweet in all_tweets if validate_tweet_data(tweet)]
        
        self.logger.info(f"Final collection: {len(valid_tweets)} valid tweets from {len(all_tweets)} total")
        
        # Save raw collected data for debugging
        if valid_tweets:
            with open("data/raw_tweets_debug.json", "w", encoding="utf-8") as f:
                json.dump(valid_tweets, f, ensure_ascii=False, indent=2, default=str)
        
        return valid_tweets
    
    async def test_connection(self) -> bool:
        """Test if the scraper can connect and fetch sample data."""
        if not await self.initialize_client():
            return False
        
        try:
            # Try to search for a simple query
            test_results = await self.client.search_tweet("#nifty50", product="Latest", count=5)
            
            if test_results:
                self.logger.info("Connection test successful - able to fetch tweets")
                return True
            else:
                self.logger.warning("Connection test failed - no results returned")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """Get statistics about the current scraping session."""
        session_duration = time.time() - self.session_start
        
        return {
            'session_duration_minutes': round(session_duration / 60, 2),
            'total_requests': self.session_requests,
            'requests_per_minute': round(self.session_requests / (session_duration / 60), 2) if session_duration > 0 else 0,
            'authenticated': self.authenticated,
            'tweets_collected': len(self.scraped_tweets)
        }