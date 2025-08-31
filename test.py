"""
Integration test script for the Twitter Market Analysis Pipeline.
Tests the twikit scraper integration with existing components.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone

# Import your pipeline components
try:
    from scraper import TwitterScraper
    from data_processor import DataProcessor  
    from signal_generator import SignalGenerator
    from config import *
    from utils import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)

async def test_scraper_integration():
    """Test the integrated twikit scraper."""
    logger = setup_logging(LOG_LEVEL)
    logger.info("=== Testing Twitter Scraper Integration ===")
    
    try:
        # Initialize scraper
        scraper = TwitterScraper()
        
        # Test connection
        logger.info("Testing connection...")
        if not await scraper.test_connection():
            logger.error("Connection test failed!")
            return False
        
        # Test single hashtag scraping
        logger.info("Testing single hashtag scraping...")
        test_tweets = await scraper.scrape_hashtag("#nifty50", 10)
        
        if not test_tweets:
            logger.error("Failed to scrape any tweets!")
            return False
        
        logger.info(f"Successfully scraped {len(test_tweets)} test tweets")
        
        # Test data processing
        logger.info("Testing data processing...")
        processor = DataProcessor()
        processed_df = processor.clean_and_process_tweets(test_tweets)
        
        if processed_df.empty:
            logger.error("Data processing failed!")
            return False
        
        logger.info(f"Successfully processed {len(processed_df)} tweets")
        
        # Test signal generation
        logger.info("Testing signal generation...")
        signal_generator = SignalGenerator()
        signals_df = signal_generator.calculate_sentiment_signals(processed_df)
        
        if signals_df.empty:
            logger.error("Signal generation failed!")
            return False
        
        logger.info(f"Successfully generated signals for {len(signals_df)} tweets")
        
        # Print sample results
        print("\n=== Sample Results ===")
        if not signals_df.empty:
            sample_tweet = signals_df.iloc[0]
            print(f"Username: {sample_tweet.get('username', 'N/A')}")
            print(f"Content: {sample_tweet.get('content', 'N/A')[:100]}...")
            print(f"Engagement: {sample_tweet.get('total_engagement', 0)}")
            print(f"Sentiment Score: {sample_tweet.get('composite_sentiment', 0):.3f}")
            print(f"Trading Signal: {sample_tweet.get('trading_signal', 'N/A')}")
        
        logger.info("=== Integration test PASSED ===")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

async def run_mini_pipeline():
    """Run a mini version of the complete pipeline."""
    logger = setup_logging(LOG_LEVEL)
    logger.info("=== Running Mini Pipeline Test ===")
    
    try:
        # Reduce targets for testing
        original_target = TARGET_TWEETS
        test_target = 50  # Small number for testing
        
        # Initialize components
        scraper = TwitterScraper()
        processor = DataProcessor()
        signal_generator = SignalGenerator()
        
        # Scrape tweets
        logger.info(f"Scraping {test_target} tweets for testing...")
        raw_tweets = []
        
        # Test with just one hashtag
        test_tweets = await scraper.scrape_hashtag("#nifty50", test_target)
        raw_tweets.extend(test_tweets)
        
        if not raw_tweets:
            logger.error("No tweets collected!")
            return False
        
        # Process tweets
        processed_df = processor.clean_and_process_tweets(raw_tweets)
        
        # Generate signals
        signals_df = signal_generator.calculate_sentiment_signals(processed_df)
        final_df = signal_generator.create_composite_signals(signals_df)
        
        # Save test results
        if not final_df.empty:
            test_path = processor.save_to_parquet(final_df, "test_results.parquet")
            logger.info(f"Test results saved to: {test_path}")
            
            # Generate summary
            summary = signal_generator.generate_market_summary(final_df)
            logger.info("Market Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("=== Mini pipeline test COMPLETED ===")
        return True
        
    except Exception as e:
        logger.error(f"Mini pipeline test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("Twitter Market Analysis - Integration Testing")
    print("=" * 50)
    
    # Run basic integration test
    test1_result = await test_scraper_integration()
    
    if test1_result:
        print("\n✅ Basic integration test PASSED")
        
        # Run mini pipeline test
        test2_result = await run_mini_pipeline()
        
        if test2_result:
            print("✅ Mini pipeline test PASSED")
            print("\n🎉 All tests passed! Your pipeline is ready to run.")
            return 0
        else:
            print("❌ Mini pipeline test FAILED")
            return 1
    else:
        print("❌ Basic integration test FAILED")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        sys.exit(1)