"""Twitter scraper for Indian stock market tweets."""

import asyncio
import aiohttp
import logging
import random
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from config import *
from utils import *

class TwitterScraper:
    """Twitter scraper with anti-bot detection and rate limiting."""
    
    def __init__(self):
        self.logger = setup_logging(LOG_LEVEL)
        self.scraped_tweets = []
        self.session_requests = 0
        self.session_start = time.time()
        
    def setup_driver(self, headless: bool = True) -> webdriver.Chrome:
        """Setup Chrome driver with anti-detection measures."""
        options = Options()
        
        if headless:
            options.add_argument('--headless')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-images')
        options.add_argument('--disable-javascript')
        options.add_argument(f'--user-agent={get_random_user_agent()}')
        
        # Random window size
        width = random.randint(1200, 1920)
        height = random.randint(800, 1080)
        options.add_argument(f'--window-size={width},{height}')
        
        try:
            driver = webdriver.Chrome(options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            self.logger.error(f"Failed to setup Chrome driver: {e}")
            raise
    
    async def scrape_with_requests(self, hashtag: str, max_tweets: int = 500) -> List[Dict[str, Any]]:
        """Scrape tweets using requests + BeautifulSoup (fallback method)."""
        tweets = []
        
        try:
            search_url = create_search_url(hashtag)
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                headers={'User-Agent': get_random_user_agent()}
            ) as session:
                
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Parse tweets from HTML structure
                        tweet_elements = soup.find_all('div', {'data-testid': 'tweet'})
                        
                        for element in tweet_elements[:max_tweets]:
                            tweet_data = self.extract_tweet_data_from_element(element, 'requests')
                            if tweet_data and self.is_recent_tweet(tweet_data.get('timestamp')):
                                tweets.append(tweet_data)
                    
                    await async_sleep_with_jitter(MIN_DELAY, MAX_DELAY)
                    
        except Exception as e:
            self.logger.error(f"Error scraping with requests for {hashtag}: {e}")
        
        return tweets
    
    def scrape_with_selenium(self, hashtag: str, max_tweets: int = 500) -> List[Dict[str, Any]]:
        """Scrape tweets using Selenium WebDriver."""
        tweets = []
        driver = None
        
        try:
            driver = self.setup_driver(headless=True)
            search_url = create_search_url(hashtag)
            
            self.logger.info(f"Navigating to: {search_url}")
            driver.get(search_url)
            
            # Random wait to mimic human behavior
            time.sleep(random.uniform(3, 7))
            
            # Scroll and collect tweets
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scrolls = 10
            
            while len(tweets) < max_tweets and scroll_attempts < max_scrolls:
                try:
                    # Find tweet elements
                    tweet_elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                    
                    for element in tweet_elements:
                        if len(tweets) >= max_tweets:
                            break
                        
                        tweet_data = self.extract_tweet_data_from_element(element, 'selenium')
                        if tweet_data and self.is_recent_tweet(tweet_data.get('timestamp')):
                            tweet_hash = generate_tweet_hash(
                                tweet_data.get('content', ''),
                                tweet_data.get('username', ''),
                                str(tweet_data.get('timestamp', ''))
                            )
                            
                            # Check for duplicates
                            if not any(t.get('hash') == tweet_hash for t in tweets):
                                tweet_data['hash'] = tweet_hash
                                tweet_data['hashtag_source'] = hashtag
                                tweets.append(tweet_data)
                    
                    # Scroll down
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(random.uniform(2, 4))
                    
                    # Check if new content loaded
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        scroll_attempts += 1
                    else:
                        scroll_attempts = 0
                        last_height = new_height
                
                except Exception as e:
                    self.logger.warning(f"Error during scrolling: {e}")
                    break
            
        except Exception as e:
            self.logger.error(f"Error scraping with Selenium for {hashtag}: {e}")
        
        finally:
            if driver:
                driver.quit()
        
        return tweets
    
    def extract_tweet_data_from_element(self, element, method: str) -> Optional[Dict[str, Any]]:
        """Extract tweet data from web element."""
        try:
            if method == 'selenium':
                # Extract using Selenium WebDriver
                username_elem = element.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"] span')
                username = safe_extract_text(username_elem).replace('@', '')
                
                content_elem = element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                content = safe_extract_text(content_elem)
                
                time_elem = element.find_element(By.CSS_SELECTOR, 'time')
                timestamp_str = time_elem.get_attribute('datetime') if time_elem else None
                
                # Extract engagement metrics
                try:
                    likes_elem = element.find_element(By.CSS_SELECTOR, '[data-testid="like"] span')
                    likes = int(safe_extract_text(likes_elem).replace(',', '') or '0')
                except:
                    likes = 0
                
                try:
                    retweets_elem = element.find_element(By.CSS_SELECTOR, '[data-testid="retweet"] span')
                    retweets = int(safe_extract_text(retweets_elem).replace(',', '') or '0')
                except:
                    retweets = 0
                
                try:
                    replies_elem = element.find_element(By.CSS_SELECTOR, '[data-testid="reply"] span')
                    replies = int(safe_extract_text(replies_elem).replace(',', '') or '0')
                except:
                    replies = 0
                
            else:  # BeautifulSoup method
                username_elem = element.find('span', {'data-testid': 'User-Name'})
                username = safe_extract_text(username_elem).replace('@', '') if username_elem else ""
                
                content_elem = element.find('div', {'data-testid': 'tweetText'})
                content = safe_extract_text(content_elem) if content_elem else ""
                
                time_elem = element.find('time')
                timestamp_str = time_elem.get('datetime') if time_elem else None
                
                # Extract engagement metrics from BeautifulSoup
                likes = retweets = replies = 0
                
                like_elem = element.find('div', {'data-testid': 'like'})
                if like_elem:
                    likes = int(safe_extract_text(like_elem).replace(',', '') or '0')
                
                retweet_elem = element.find('div', {'data-testid': 'retweet'})
                if retweet_elem:
                    retweets = int(safe_extract_text(retweet_elem).replace(',', '') or '0')
                
                reply_elem = element.find('div', {'data-testid': 'reply'})
                if reply_elem:
                    replies = int(safe_extract_text(reply_elem).replace(',', '') or '0')
            
            # Parse timestamp
            timestamp = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    timestamp = parse_relative_time(timestamp_str)
            
            if not all([username, content]):
                return None
            
            return {
                'username': username,
                'content': content,
                'timestamp': timestamp or datetime.now(timezone.utc),
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'hashtags': extract_hashtags(content),
                'mentions': extract_mentions(content),
                'scraped_at': datetime.now(timezone.utc)
            }
        
        except Exception as e:
            self.logger.warning(f"Error extracting tweet data: {e}")
            return None
    
    def is_recent_tweet(self, timestamp: Optional[datetime]) -> bool:
        """Check if tweet is within the target time window."""
        if not timestamp:
            return False
        
        age_hours = calculate_tweet_age_hours(timestamp)
        return age_hours <= TIME_WINDOW_HOURS
    
    async def scrape_hashtag(self, hashtag: str, max_tweets_per_hashtag: int) -> List[Dict[str, Any]]:
        """Scrape tweets for a specific hashtag."""
        self.logger.info(f"Starting to scrape hashtag: {hashtag}")
        
        # Try Selenium first, fall back to requests if needed
        tweets = []
        
        try:
            # Primary method: Selenium
            tweets = self.scrape_with_selenium(hashtag, max_tweets_per_hashtag)
            
            if len(tweets) < max_tweets_per_hashtag // 2:
                # Fallback method: Requests + BeautifulSoup
                self.logger.info(f"Low tweet count for {hashtag}, trying fallback method")
                additional_tweets = await self.scrape_with_requests(hashtag, max_tweets_per_hashtag)
                tweets.extend(additional_tweets)
        
        except Exception as e:
            self.logger.error(f"Error scraping hashtag {hashtag}: {e}")
        
        # Deduplicate and validate
        tweets = deduplicate_tweets(tweets)
        tweets = [t for t in tweets if validate_tweet_data(t)]
        
        self.logger.info(f"Collected {len(tweets)} tweets for hashtag: {hashtag}")
        return tweets
    
    async def scrape_all_hashtags(self) -> List[Dict[str, Any]]:
        """Scrape tweets from all configured hashtags."""
        self.logger.info("Starting comprehensive tweet scraping")
        
        all_tweets = []
        tweets_per_hashtag = TARGET_TWEETS // len(SEARCH_HASHTAGS)
        
        # Create semaphore for concurrent scraping
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def scrape_with_semaphore(hashtag):
            async with semaphore:
                return await self.scrape_hashtag(hashtag, tweets_per_hashtag)
        
        # Scrape all hashtags concurrently
        tasks = [scrape_with_semaphore(hashtag) for hashtag in SEARCH_HASHTAGS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to scrape {SEARCH_HASHTAGS[i]}: {result}")
            else:
                all_tweets.extend(result)
        
        # Final deduplication across all hashtags
        all_tweets = deduplicate_tweets(all_tweets)
        
        self.logger.info(f"Total tweets collected: {len(all_tweets)}")
        return all_tweets