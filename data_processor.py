"""Data processing and storage for Twitter scraping pipeline."""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
import fastparquet as fp

from config import *
from utils import setup_logging, clean_text, deduplicate_tweets, validate_tweet_data

class DataProcessor:
    """Handle data processing, cleaning, and storage operations."""
    
    def __init__(self):
        self.logger = setup_logging(LOG_LEVEL)
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        Path(DATA_DIR).mkdir(exist_ok=True)
        self.logger.info(f"Data directory ensured: {DATA_DIR}")
    
    def clean_and_process_tweets(self, raw_tweets: List[Dict[str, Any]]) -> pd.DataFrame:
        """Clean and process raw tweet data."""
        if not raw_tweets:
            self.logger.warning("No tweets to process")
            return pd.DataFrame()
        
        self.logger.info(f"Processing {len(raw_tweets)} raw tweets")
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_tweets)
        
        # Data validation and cleaning
        initial_count = len(df)
        df = df[df.apply(lambda x: validate_tweet_data(x.to_dict()), axis=1)]
        self.logger.info(f"Removed {initial_count - len(df)} invalid tweets")
        
        if df.empty:
            return df
        
        # Clean text content
        df['content_cleaned'] = df['content'].apply(clean_text)
        df['content_length'] = df['content'].str.len()
        df['cleaned_length'] = df['content_cleaned'].str.len()
        
        # Process timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Calculate engagement metrics
        df['total_engagement'] = df['likes'] + df['retweets'] + df['replies']
        df['engagement_rate'] = df['total_engagement'] / (df['content_length'] + 1)  # Avoid division by zero
        
        # Process hashtags and mentions
        df['hashtag_count'] = df['hashtags'].apply(len)
        df['mention_count'] = df['mentions'].apply(len)
        df['hashtags_str'] = df['hashtags'].apply(lambda x: ','.join(x) if x else '')
        df['mentions_str'] = df['mentions'].apply(lambda x: ','.join(x) if x else '')
        
        # Market sentiment indicators
        df['has_market_keywords'] = df['content_cleaned'].apply(self.contains_market_keywords)
        df['sentiment_score'] = df['content_cleaned'].apply(self.calculate_basic_sentiment)
        
        # Data quality metrics
        df['is_potential_bot'] = df.apply(self.detect_potential_bot, axis=1)
        df['quality_score'] = df.apply(self.calculate_quality_score, axis=1)
        
        # Filter out low-quality tweets
        quality_threshold = 0.3
        high_quality_df = df[df['quality_score'] >= quality_threshold].copy()
        
        self.logger.info(f"Processed tweets: {len(df)} -> {len(high_quality_df)} high quality")
        
        return high_quality_df
    
    def contains_market_keywords(self, text: str) -> bool:
        """Check if text contains market-related keywords."""
        if not text:
            return False
        
        market_keywords = {
            'nifty', 'sensex', 'intraday', 'banknifty', 'stock', 'market',
            'trading', 'trader', 'investment', 'profit', 'loss', 'buy', 'sell',
            'bull', 'bear', 'trend', 'analysis', 'chart', 'support', 'resistance'
        }
        
        text_words = set(text.lower().split())
        return bool(market_keywords.intersection(text_words))
    
    def calculate_basic_sentiment(self, text: str) -> float:
        """Calculate basic sentiment score (-1 to 1)."""
        if not text:
            return 0.0
        
        positive_words = {
            'bull', 'bullish', 'up', 'rise', 'gain', 'profit', 'green', 'high',
            'moon', 'rocket', 'surge', 'rally', 'breakout', 'strong', 'good'
        }
        
        negative_words = {
            'bear', 'bearish', 'down', 'fall', 'loss', 'red', 'low', 'crash',
            'dump', 'drop', 'weak', 'bad', 'sell', 'short', 'resistance'
        }
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment))  # Clamp between -1 and 1
    
    def detect_potential_bot(self, row: pd.Series) -> bool:
        """Detect potential bot accounts based on patterns."""
        username = row.get('username', '')
        content = row.get('content', '')
        engagement = row.get('total_engagement', 0)
        content_length = row.get('content_length', 0)
        
        # Bot detection heuristics
        bot_indicators = 0
        
        # Username patterns
        if len(username) > 15 or any(char.isdigit() for char in username[-3:]):
            bot_indicators += 1
        
        # Content patterns
        if content_length < 10 or content_length > 280:
            bot_indicators += 1
        
        # Engagement anomalies (very high engagement for short content)
        if content_length > 0 and (engagement / content_length) > 5:
            bot_indicators += 1
        
        # Repetitive content indicators
        words = content.split()
        if len(set(words)) < len(words) * 0.5 and len(words) > 5:  # High repetition
            bot_indicators += 1
        
        return bot_indicators >= 2
    
    def calculate_quality_score(self, row: pd.Series) -> float:
        """Calculate overall quality score for a tweet."""
        score = 1.0
        
        # Penalize potential bots
        if row.get('is_potential_bot', False):
            score *= 0.5
        
        # Reward market relevance
        if row.get('has_market_keywords', False):
            score *= 1.2
        
        # Reward engagement
        engagement = row.get('total_engagement', 0)
        if engagement > 10:
            score *= 1.1
        elif engagement > 50:
            score *= 1.3
        
        # Penalize very short or very long content
        content_length = row.get('content_length', 0)
        if content_length < 20 or content_length > 250:
            score *= 0.8
        
        # Reward reasonable hashtag usage
        hashtag_count = row.get('hashtag_count', 0)
        if 1 <= hashtag_count <= 5:
            score *= 1.1
        elif hashtag_count > 8:
            score *= 0.7
        
        return min(1.0, score)  # Cap at 1.0
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str, compression: str = 'snappy') -> str:
        """Save DataFrame to Parquet format."""
        if df.empty:
            self.logger.warning(f"Empty DataFrame, not saving {filename}")
            return ""
        
        filepath = os.path.join(DATA_DIR, filename)
        
        try:
            # Optimize data types for storage
            df_optimized = self.optimize_dataframe_dtypes(df.copy())
            
            # Save to Parquet
            df_optimized.to_parquet(
                filepath,
                compression=compression,
                index=False,
                engine='fastparquet'
            )
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            self.logger.info(f"Saved {len(df_optimized)} records to {filepath} ({file_size:.2f} MB)")
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error saving to Parquet: {e}")
            return ""
    
    def optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for storage efficiency."""
        # Convert integers to smallest possible type
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert floats to float32 where possible
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category where beneficial
        for col in df.select_dtypes(include=['object']).columns:
            if col in ['day_of_week', 'hashtag_source']:
                df[col] = df[col].astype('category')
        
        return df
    
    def load_from_parquet(self, filename: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from Parquet file."""
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"File not found: {filepath}")
            return None
        
        try:
            df = pd.read_parquet(filepath, engine='fastparquet')
            self.logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading from Parquet: {e}")
            return None
    
    def merge_with_existing_data(self, new_df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Merge new data with existing data, handling duplicates."""
        existing_df = self.load_from_parquet(filename)
        
        if existing_df is None or existing_df.empty:
            return new_df
        
        # Combine datasets
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates based on hash
        if 'hash' in combined_df.columns:
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['hash'], keep='last')
            removed_count = initial_count - len(combined_df)
            self.logger.info(f"Removed {removed_count} duplicate records")
        
        # Sort by timestamp
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp', ascending=False)
        
        return combined_df
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data statistics."""
        if df.empty:
            return {}
        
        stats = {
            'total_tweets': len(df),
            'unique_users': df['username'].nunique() if 'username' in df else 0,
            'date_range': {
                'start': df['timestamp'].min().isoformat() if 'timestamp' in df else None,
                'end': df['timestamp'].max().isoformat() if 'timestamp' in df else None
            },
            'engagement': {
                'total_likes': int(df['likes'].sum()) if 'likes' in df else 0,
                'total_retweets': int(df['retweets'].sum()) if 'retweets' in df else 0,
                'total_replies': int(df['replies'].sum()) if 'replies' in df else 0,
                'avg_engagement_rate': float(df['engagement_rate'].mean()) if 'engagement_rate' in df else 0
            },
            'content': {
                'avg_length': float(df['content_length'].mean()) if 'content_length' in df else 0,
                'avg_hashtags': float(df['hashtag_count'].mean()) if 'hashtag_count' in df else 0,
                'avg_mentions': float(df['mention_count'].mean()) if 'mention_count' in df else 0
            },
            'quality': {
                'avg_quality_score': float(df['quality_score'].mean()) if 'quality_score' in df else 0,
                'potential_bots': int(df['is_potential_bot'].sum()) if 'is_potential_bot' in df else 0,
                'market_relevant': int(df['has_market_keywords'].sum()) if 'has_market_keywords' in df else 0
            },
            'sentiment': {
                'avg_sentiment': float(df['sentiment_score'].mean()) if 'sentiment_score' in df else 0,
                'positive_tweets': int((df['sentiment_score'] > 0.1).sum()) if 'sentiment_score' in df else 0,
                'negative_tweets': int((df['sentiment_score'] < -0.1).sum()) if 'sentiment_score' in df else 0,
                'neutral_tweets': int((df['sentiment_score'].abs() <= 0.1).sum()) if 'sentiment_score' in df else 0
            }
        }
        
        return stats