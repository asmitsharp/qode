import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timezone
import re

from config import *
from utils import setup_logging, market_sentiment_keywords

class SignalGenerator:
    """Generate quantitative trading signals from tweet text data."""
    
    def __init__(self):
        self.logger = setup_logging(LOG_LEVEL)
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.sentiment_keywords = market_sentiment_keywords()
        
    def prepare_text_data(self, df: pd.DataFrame) -> List[str]:
        """Prepare and clean text data for TF-IDF processing."""
        if 'content_cleaned' not in df.columns:
            self.logger.error("content_cleaned column not found")
            return []
        
        # Filter out very short texts
        texts = df[df['content_cleaned'].str.len() >= 10]['content_cleaned'].tolist()
        
        # Additional cleaning for TF-IDF
        cleaned_texts = []
        for text in texts:
            # Remove extra whitespace and special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip().lower()
            
            if len(text.split()) >= 3:  # At least 3 words
                cleaned_texts.append(text)
        
        self.logger.info(f"Prepared {len(cleaned_texts)} texts for TF-IDF processing")
        return cleaned_texts
    
    def create_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF feature matrix."""
        if not texts:
            self.logger.error("No texts provided for TF-IDF")
            return np.array([])
        
        # Initialize TF-IDF vectorizer with market-specific settings
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words=list(STOP_WORDS_CUSTOM),
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.logger.info(f"Created TF-IDF matrix: {tfidf_matrix.shape}")
            return tfidf_matrix.toarray()
        
        except Exception as e:
            self.logger.error(f"Error creating TF-IDF features: {e}")
            return np.array([])
    
    def calculate_sentiment_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various sentiment-based trading signals."""
        if df.empty:
            return df
        
        df_signals = df.copy()
        
        # Enhanced sentiment scoring
        df_signals['bullish_score'] = df_signals['content_cleaned'].apply(self.calculate_bullish_score)
        df_signals['bearish_score'] = df_signals['content_cleaned'].apply(self.calculate_bearish_score)
        df_signals['volatility_score'] = df_signals['content_cleaned'].apply(self.calculate_volatility_score)
        
        # Composite sentiment signal
        df_signals['composite_sentiment'] = (
            df_signals['bullish_score'] - df_signals['bearish_score'] + 
            df_signals['sentiment_score']
        ) / 3
        
        # Volume-weighted sentiment (using engagement as proxy for volume)
        df_signals['weighted_sentiment'] = (
            df_signals['composite_sentiment'] * 
            np.log1p(df_signals['total_engagement'])
        )
        
        # Time-decay factor (recent tweets get higher weight)
        hours_ago = (datetime.now(timezone.utc) - df_signals['timestamp']).dt.total_seconds() / 3600
        df_signals['time_decay'] = np.exp(-hours_ago / 12)  # Decay over 12 hours
        df_signals['time_weighted_sentiment'] = df_signals['composite_sentiment'] * df_signals['time_decay']
        
        return df_signals
    
    def calculate_bullish_score(self, text: str) -> float:
        """Calculate bullish sentiment score."""
        if not text:
            return 0.0
        
        bullish_patterns = [
            r'\b(moon|rocket|surge|rally|breakout|pump)\b',
            r'\b(strong\s+buy|bullish\s+trend|upward\s+momentum)\b',
            r'\b(resistance\s+broken|support\s+holding)\b',
            r'\b(target\s+achieved|profit\s+booking)\b'
        ]
        
        score = 0.0
        for pattern in bullish_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.2
        
        # Keyword-based scoring
        for keyword in self.sentiment_keywords['bullish']:
            if keyword in text:
                score += 0.1
        
        return min(1.0, score)
    
    def calculate_bearish_score(self, text: str) -> float:
        """Calculate bearish sentiment score."""
        if not text:
            return 0.0
        
        bearish_patterns = [
            r'\b(crash|dump|fall|drop|correction)\b',
            r'\b(strong\s+sell|bearish\s+trend|downward\s+pressure)\b',
            r'\b(support\s+broken|resistance\s+holding)\b',
            r'\b(stop\s+loss|cut\s+losses)\b'
        ]
        
        score = 0.0
        for pattern in bearish_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.2
        
        # Keyword-based scoring
        for keyword in self.sentiment_keywords['bearish']:
            if keyword in text:
                score += 0.1
        
        return min(1.0, score)
    
    def calculate_volatility_score(self, text: str) -> float:
        """Calculate expected volatility score."""
        if not text:
            return 0.0
        
        volatility_keywords = [
            'volatile', 'swing', 'range', 'whipsaw', 'choppy',
            'uncertain', 'unclear', 'mixed', 'conflicted'
        ]
        
        score = 0.0
        for keyword in volatility_keywords:
            if keyword in text:
                score += 0.15
        
        # High engagement often indicates volatile market conditions
        return min(1.0, score)
    
    def generate_tfidf_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using TF-IDF analysis."""
        if df.empty:
            return df
        
        df_with_signals = df.copy()
        
        # Prepare texts
        texts = self.prepare_text_data(df_with_signals)
        
        if len(texts) < 10:  # Need minimum texts for meaningful analysis
            self.logger.warning("Insufficient text data for TF-IDF analysis")
            return df_with_signals
        
        # Create TF-IDF features
        tfidf_features = self.create_tfidf_features(texts)
        
        if tfidf_features.size == 0:
            return df_with_signals
        
        # Dimensionality reduction
        tfidf_reduced = self.svd.fit_transform(tfidf_features)
        tfidf_scaled = self.scaler.fit_transform(tfidf_reduced)
        
        # Clustering to identify market themes
        n_clusters = min(5, len(texts) // 20)  # Dynamic cluster count
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_scaled)
            
            # Add cluster information back to dataframe
            valid_indices = df_with_signals[df_with_signals['content_cleaned'].str.len() >= 10].index
            if len(valid_indices) == len(clusters):
                cluster_series = pd.Series(index=df_with_signals.index, dtype='int64')
                cluster_series.loc[valid_indices] = clusters
                df_with_signals['topic_cluster'] = cluster_series.fillna(-1)
            
        # Calculate TF-IDF based signals
        if len(valid_indices) == len(tfidf_scaled):
            # Market momentum signal (based on first principal component)
            momentum_series = pd.Series(index=df_with_signals.index, dtype='float64')
            momentum_series.loc[valid_indices] = tfidf_scaled[:, 0]
            df_with_signals['tfidf_momentum'] = momentum_series.fillna(0)
            
            # Market attention signal (based on TF-IDF magnitude)
            attention_scores = np.linalg.norm(tfidf_features, axis=1)
            attention_series = pd.Series(index=df_with_signals.index, dtype='float64')
            attention_series.loc[valid_indices] = attention_scores
            df_with_signals['market_attention'] = attention_series.fillna(0)
        
        return df_with_signals
    
    def create_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite trading signals."""
        if df.empty:
            return df
        
        df_final = df.copy()
        
        # Normalize signals to [-1, 1] range
        signal_columns = [
            'composite_sentiment', 'weighted_sentiment', 'time_weighted_sentiment',
            'tfidf_momentum', 'bullish_score', 'bearish_score'
        ]
        
        available_columns = [col for col in signal_columns if col in df_final.columns]
        
        if not available_columns:
            self.logger.warning("No signal columns available for composite signal creation")
            return df_final
        
        # Create composite signals
        df_final['primary_signal'] = 0.0
        df_final['confidence_score'] = 0.0
        
        for col in available_columns:
            # Normalize to [-1, 1]
            col_min, col_max = df_final[col].min(), df_final[col].max()
            if col_max != col_min:
                normalized = 2 * (df_final[col] - col_min) / (col_max - col_min) - 1
                df_final[f'{col}_normalized'] = normalized
            else:
                df_final[f'{col}_normalized'] = 0.0
        
        # Weight different signals
        weights = {
            'composite_sentiment': 0.3,
            'weighted_sentiment': 0.25,
            'time_weighted_sentiment': 0.2,
            'tfidf_momentum': 0.15,
            'bullish_score': 0.05,
            'bearish_score': 0.05
        }
        
        for col, weight in weights.items():
            norm_col = f'{col}_normalized'
            if norm_col in df_final.columns:
                df_final['primary_signal'] += weight * df_final[norm_col]
        
        # Calculate confidence intervals
        df_final['signal_strength'] = abs(df_final['primary_signal'])
        df_final['signal_direction'] = np.sign(df_final['primary_signal'])
        
        # Confidence based on consistency across signals and engagement
        signal_consistency = df_final[[f'{col}_normalized' for col in available_columns 
                                     if f'{col}_normalized' in df_final.columns]].std(axis=1)
        df_final['confidence_score'] = (
            (1 - signal_consistency) * 
            np.log1p(df_final['total_engagement']) / np.log1p(df_final['total_engagement'].max() + 1)
        )
        
        # Trading recommendations
        df_final['trading_signal'] = 'HOLD'
        df_final.loc[
            (df_final['primary_signal'] > SIGNAL_THRESHOLD_POSITIVE) & 
            (df_final['confidence_score'] > 0.5), 'trading_signal'
        ] = 'BUY'
        df_final.loc[
            (df_final['primary_signal'] < SIGNAL_THRESHOLD_NEGATIVE) & 
            (df_final['confidence_score'] > 0.5), 'trading_signal'
        ] = 'SELL'
        
        return df_final
    
    def aggregate_signals_by_time(self, df: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
        """Aggregate signals by time periods for trend analysis."""
        if df.empty or 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        # Set timestamp as index for resampling
        df_time = df.set_index('timestamp')
        
        # Aggregation functions for different columns
        agg_funcs = {
            'primary_signal': 'mean',
            'confidence_score': 'mean',
            'signal_strength': 'mean',
            'composite_sentiment': 'mean',
            'total_engagement': 'sum',
            'likes': 'sum',
            'retweets': 'sum',
            'replies': 'sum',
            'username': 'count'  # Tweet count per period
        }
        
        # Filter aggregation functions to available columns
        available_agg = {k: v for k, v in agg_funcs.items() if k in df_time.columns}
        
        if not available_agg:
            return pd.DataFrame()
        
        # Resample and aggregate
        aggregated = df_time.resample(freq).agg(available_agg)
        aggregated.rename(columns={'username': 'tweet_count'}, inplace=True)
        
        # Calculate rolling trends
        if len(aggregated) > 3:
            aggregated['signal_trend'] = aggregated['primary_signal'].rolling(window=3, center=True).mean()
            aggregated['momentum_change'] = aggregated['primary_signal'].diff()
            
        # Reset index to make timestamp a column again
        aggregated.reset_index(inplace=True)
        
        return aggregated
    
    def get_top_influential_tweets(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Get most influential tweets based on signals and engagement."""
        if df.empty:
            return df
        
        # Calculate influence score
        influence_factors = []
        
        if 'signal_strength' in df.columns:
            influence_factors.append(df['signal_strength'])
        
        if 'confidence_score' in df.columns:
            influence_factors.append(df['confidence_score'])
        
        if 'total_engagement' in df.columns:
            normalized_engagement = df['total_engagement'] / (df['total_engagement'].max() + 1)
            influence_factors.append(normalized_engagement)
        
        if 'quality_score' in df.columns:
            influence_factors.append(df['quality_score'])
        
        if not influence_factors:
            return df.head(n)
        
        # Combine influence factors
        df['influence_score'] = sum(influence_factors) / len(influence_factors)
        
        # Get top influential tweets
        top_tweets = df.nlargest(n, 'influence_score')
        
        return top_tweets[['username', 'content', 'timestamp', 'trading_signal', 
                         'primary_signal', 'confidence_score', 'influence_score',
                         'likes', 'retweets', 'replies']].copy()
    
    def generate_market_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive market sentiment summary."""
        if df.empty:
            return {}
        
        signal_columns = ['primary_signal', 'confidence_score', 'composite_sentiment']
        available_signals = [col for col in signal_columns if col in df.columns]
        
        if not available_signals:
            return {}
        
        summary = {
            'overall_sentiment': {
                'mean_signal': float(df['primary_signal'].mean()) if 'primary_signal' in df else 0,
                'signal_std': float(df['primary_signal'].std()) if 'primary_signal' in df else 0,
                'bullish_tweets': int((df['primary_signal'] > 0.1).sum()) if 'primary_signal' in df else 0,
                'bearish_tweets': int((df['primary_signal'] < -0.1).sum()) if 'primary_signal' in df else 0,
                'neutral_tweets': int((abs(df['primary_signal']) <= 0.1).sum()) if 'primary_signal' in df else 0
            },
            'confidence_metrics': {
                'avg_confidence': float(df['confidence_score'].mean()) if 'confidence_score' in df else 0,
                'high_confidence_signals': int((df['confidence_score'] > 0.7).sum()) if 'confidence_score' in df else 0
            },
            'trading_recommendations': {
                'buy_signals': int((df['trading_signal'] == 'BUY').sum()) if 'trading_signal' in df else 0,
                'sell_signals': int((df['trading_signal'] == 'SELL').sum()) if 'trading_signal' in df else 0,
                'hold_signals': int((df['trading_signal'] == 'HOLD').sum()) if 'trading_signal' in df else 0
            },
            'market_themes': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Topic analysis if available
        if 'topic_cluster' in df.columns:
            topic_counts = df['topic_cluster'].value_counts().head(5)
            summary['market_themes'] = {f'theme_{k}': int(v) for k, v in topic_counts.items() if k != -1}
        
        return summary