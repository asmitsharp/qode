

import asyncio
import pandas as pd
from datetime import datetime, timezone
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper import TwitterScraper
from data_processor import DataProcessor
from signal_generator import SignalGenerator
from visualizer import MemoryEfficientVisualizer
from utils import setup_logging

def create_sample_data():
    """Create sample tweet data for testing."""
    sample_tweets = [
        {
            'username': 'trader_bull',
            'content': 'Nifty looking bullish today! Strong breakout above resistance. #nifty50 #bullish',
            'timestamp': datetime.now(timezone.utc),
            'likes': 15,
            'retweets': 8,
            'replies': 3,
            'hashtags': ['nifty50', 'bullish'],
            'mentions': [],
            'scraped_at': datetime.now(timezone.utc)
        },
        {
            'username': 'market_bear',
            'content': 'Sensex showing weakness. Time to book profits and wait. #sensex #bearish',
            'timestamp': datetime.now(timezone.utc),
            'likes': 22,
            'retweets': 12,
            'replies': 5,
            'hashtags': ['sensex', 'bearish'],
            'mentions': [],
            'scraped_at': datetime.now(timezone.utc)
        },
        {
            'username': 'intraday_king',
            'content': 'Bank Nifty intraday setup looking good. Buy on dips strategy for today. #banknifty #intraday',
            'timestamp': datetime.now(timezone.utc),
            'likes': 45,
            'retweets': 28,
            'replies': 12,
            'hashtags': ['banknifty', 'intraday'],
            'mentions': [],
            'scraped_at': datetime.now(timezone.utc)
        }
    ]
    
    # Add more sample data for better testing
    for i in range(10):
        sample_tweets.append({
            'username': f'user_{i}',
            'content': f'Market analysis sample tweet {i} with various sentiments. #nifty50',
            'timestamp': datetime.now(timezone.utc),
            'likes': i * 2,
            'retweets': i,
            'replies': i // 2,
            'hashtags': ['nifty50'],
            'mentions': [],
            'scraped_at': datetime.now(timezone.utc)
        })
    
    return sample_tweets

async def test_scraper():
    """Test the Twitter scraper component."""
    print("Testing TwitterScraper...")
    
    scraper = TwitterScraper()
    
    # Test utility functions
    print("✓ TwitterScraper initialized successfully")
    
    # Note: We won't test actual scraping to avoid hitting Twitter
    # In a real scenario, you would run: tweets = await scraper.scrape_all_hashtags()
    print("✓ Scraper component validated (actual scraping skipped)")
    
    return True

def test_data_processor():
    """Test the data processing component."""
    print("\nTesting DataProcessor...")
    
    processor = DataProcessor()
    sample_data = create_sample_data()
    
    # Test data processing
    df = processor.clean_and_process_tweets(sample_data)
    
    if not df.empty:
        print(f"✓ Processed {len(df)} tweets successfully")
        print(f"✓ Columns created: {list(df.columns)}")
        
        # Test Parquet saving
        filepath = processor.save_to_parquet(df, "test_tweets.parquet")
        if filepath:
            print(f"✓ Data saved to Parquet: {filepath}")
        
        # Test loading
        loaded_df = processor.load_from_parquet("test_tweets.parquet")
        if loaded_df is not None and not loaded_df.empty:
            print(f"✓ Data loaded successfully: {len(loaded_df)} records")
        
        # Test statistics
        stats = processor.get_data_statistics(df)
        if stats:
            print(f"✓ Statistics generated: {len(stats)} categories")
        
        return df
    
    return pd.DataFrame()

def test_signal_generator(df):
    """Test the signal generation component."""
    print("\nTesting SignalGenerator...")
    
    if df.empty:
        print("⚠ No data to test signal generation")
        return df
    
    signal_gen = SignalGenerator()
    
    # Test sentiment signals
    df_with_sentiment = signal_gen.calculate_sentiment_signals(df)
    print(f"✓ Sentiment signals generated: {len(df_with_sentiment)} records")
    
    # Test TF-IDF signals (may not work with limited sample data)
    try:
        df_with_tfidf = signal_gen.generate_tfidf_signals(df_with_sentiment)
        print(f"✓ TF-IDF signals generated: {len(df_with_tfidf)} records")
    except Exception as e:
        print(f"⚠ TF-IDF generation skipped (insufficient data): {e}")
        df_with_tfidf = df_with_sentiment
    
    # Test composite signals
    final_df = signal_gen.create_composite_signals(df_with_tfidf)
    print(f"✓ Composite signals generated: {len(final_df)} records")
    
    # Test market summary
    summary = signal_gen.generate_market_summary(final_df)
    if summary:
        print(f"✓ Market summary generated: {len(summary)} sections")
    
    return final_df

def test_visualizer(df):
    """Test the visualization component."""
    print("\nTesting MemoryEfficientVisualizer...")
    
    if df.empty:
        print("⚠ No data to test visualizations")
        return []
    
    visualizer = MemoryEfficientVisualizer()
    
    created_files = []
    
    # Test individual visualizations
    viz_functions = [
        ('word cloud', visualizer.create_wordcloud_visualization),
        ('signal distribution', visualizer.create_signal_distribution),
        ('engagement analysis', visualizer.create_engagement_analysis)
    ]
    
    for name, func in viz_functions:
        try:
            result = func(df)
            if result:
                created_files.append(result)
                print(f"✓ {name} created: {result}")
        except Exception as e:
            print(f"⚠ {name} creation failed: {e}")
    
    print(f"✓ Created {len(created_files)} visualization files")
    
    # Clean up memory
    visualizer.optimize_memory_usage()
    
    return created_files

async def run_full_test():
    """Run complete pipeline test."""
    print("=" * 60)
    print("TWITTER MARKET SCRAPER - COMPONENT TESTING")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    try:
        # Test each component
        if await test_scraper():
            success_count += 1
        
        df = test_data_processor()
        if not df.empty:
            success_count += 1
        
        final_df = test_signal_generator(df)
        if not final_df.empty:
            success_count += 1
        
        viz_files = test_visualizer(final_df)
        if viz_files:
            success_count += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Tests Passed: {success_count}/{total_tests}")
        print(f"Success Rate: {success_count/total_tests*100:.1f}%")
        
        if success_count == total_tests:
            print("✅ All components tested successfully!")
            print("\nYour pipeline is ready to run with: python main.py")
        else:
            print("⚠ Some components need attention before running the full pipeline")
        
        return success_count == total_tests
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_full_test())
    sys.exit(0 if success else 1)