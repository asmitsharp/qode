"""Main execution script for Twitter market analysis scraper."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

from scraper import TwitterScraper
from data_processor import DataProcessor
from signal_generator import SignalGenerator
from visualizer import MemoryEfficientVisualizer
from config import *
from utils import setup_logging

class MarketAnalysisPipeline:
    """Complete pipeline for Twitter market analysis."""
    
    def __init__(self):
        self.logger = setup_logging(LOG_LEVEL)
        self.scraper = TwitterScraper()
        self.processor = DataProcessor()
        self.signal_generator = SignalGenerator()
        self.visualizer = MemoryEfficientVisualizer()
        
    async def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        start_time = time.time()
        self.logger.info("=== Starting Twitter Market Analysis Pipeline ===")
        
        results = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'running',
            'tweets_collected': 0,
            'tweets_processed': 0,
            'signals_generated': 0,
            'visualizations_created': 0,
            'errors': []
        }
        
        try:
            # Step 1: Scrape tweets
            self.logger.info("Step 1: Scraping tweets from Twitter")
            raw_tweets = await self.scraper.scrape_all_hashtags()
            
            if not raw_tweets:
                raise Exception("No tweets collected from scraping")
            
            results['tweets_collected'] = len(raw_tweets)
            self.logger.info(f"Collected {len(raw_tweets)} raw tweets")
            
            # Step 2: Process and clean data
            self.logger.info("Step 2: Processing and cleaning tweet data")
            processed_df = self.processor.clean_and_process_tweets(raw_tweets)
            
            if processed_df.empty:
                raise Exception("No tweets remained after processing")
            
            results['tweets_processed'] = len(processed_df)
            
            # Save raw data
            raw_path = self.processor.save_to_parquet(processed_df, RAW_DATA_FILE)
            if raw_path:
                self.logger.info(f"Raw data saved to: {raw_path}")
            
            # Step 3: Generate trading signals
            self.logger.info("Step 3: Generating trading signals")
            
            # Calculate sentiment signals
            signals_df = self.signal_generator.calculate_sentiment_signals(processed_df)
            
            # Generate TF-IDF signals
            signals_df = self.signal_generator.generate_tfidf_signals(signals_df)
            
            # Create composite signals
            final_signals_df = self.signal_generator.create_composite_signals(signals_df)
            
            results['signals_generated'] = len(final_signals_df)
            
            # Save processed signals
            signals_path = self.processor.save_to_parquet(final_signals_df, SIGNALS_DATA_FILE)
            if signals_path:
                self.logger.info(f"Signals saved to: {signals_path}")
            
            # Step 4: Generate insights and aggregations
            self.logger.info("Step 4: Generating market insights")
            
            # Time-based aggregations
            hourly_signals = self.signal_generator.aggregate_signals_by_time(final_signals_df, '1H')
            
            # Get influential tweets
            top_tweets = self.signal_generator.get_top_influential_tweets(final_signals_df, 50)
            
            # Generate market summary
            market_summary = self.signal_generator.generate_market_summary(final_signals_df)
            
            # Data statistics
            data_stats = self.processor.get_data_statistics(final_signals_df)
            
            # Step 5: Create visualizations
            self.logger.info("Step 5: Creating visualizations")
            
            visualization_files = self.visualizer.create_comprehensive_report(
                final_signals_df, 
                {**market_summary, **data_stats}
            )
            
            results['visualizations_created'] = len(visualization_files)
            
            # Step 6: Generate final report
            self.logger.info("Step 6: Generating final analysis report")
            
            report_data = {
                'execution_summary': results,
                'market_summary': market_summary,
                'data_statistics': data_stats,
                'top_influential_tweets': top_tweets.to_dict('records') if not top_tweets.empty else [],
                'hourly_trends': hourly_signals.to_dict('records') if not hourly_signals.empty else [],
                'visualization_files': visualization_files
            }
            
            # Save final report
            self.save_analysis_report(report_data)
            
            # Update results
            execution_time = time.time() - start_time
            results.update({
                'status': 'completed',
                'execution_time_seconds': round(execution_time, 2),
                'end_time': datetime.now(timezone.utc).isoformat(),
                'final_dataset_size': len(final_signals_df),
                'data_files_created': [raw_path, signals_path] if raw_path and signals_path else [],
                'visualization_files': visualization_files
            })
            
            self.logger.info(f"=== Analysis completed successfully in {execution_time:.2f} seconds ===")
            
            # Print summary
            self.print_execution_summary(results, market_summary)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            results['status'] = 'failed'
            results['errors'].append(str(e))
            results['end_time'] = datetime.now(timezone.utc).isoformat()
            return results
        
        finally:
            # Cleanup
            self.visualizer.optimize_memory_usage()
    
    def save_analysis_report(self, report_data: Dict[str, Any]) -> str:
        """Save comprehensive analysis report."""
        report_path = "data/analysis_report.json"
        
        try:
            import json
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Analysis report saved to: {report_path}")
            return report_path
        
        except Exception as e:
            self.logger.error(f"Failed to save analysis report: {e}")
            return ""
    
    def print_execution_summary(self, results: Dict[str, Any], market_summary: Dict[str, Any]):
        """Print execution summary to console."""
        print("\n" + "="*60)
        print("TWITTER MARKET ANALYSIS - EXECUTION SUMMARY")
        print("="*60)
        
        print(f"Status: {results['status'].upper()}")
        print(f"Execution Time: {results.get('execution_time_seconds', 0):.2f} seconds")
        print(f"Tweets Collected: {results['tweets_collected']:,}")
        print(f"Tweets Processed: {results['tweets_processed']:,}")
        print(f"Signals Generated: {results['signals_generated']:,}")
        print(f"Visualizations Created: {results['visualizations_created']}")
        
        if market_summary and 'overall_sentiment' in market_summary:
            sentiment = market_summary['overall_sentiment']
            print(f"\nMARKET SENTIMENT:")
            print(f"  Mean Signal: {sentiment.get('mean_signal', 0):.3f}")
            print(f"  Bullish Tweets: {sentiment.get('bullish_tweets', 0):,}")
            print(f"  Bearish Tweets: {sentiment.get('bearish_tweets', 0):,}")
            print(f"  Neutral Tweets: {sentiment.get('neutral_tweets', 0):,}")
        
        if market_summary and 'trading_recommendations' in market_summary:
            recommendations = market_summary['trading_recommendations']
            print(f"\nTRADING SIGNALS:")
            print(f"  BUY Signals: {recommendations.get('buy_signals', 0):,}")
            print(f"  SELL Signals: {recommendations.get('sell_signals', 0):,}")
            print(f"  HOLD Signals: {recommendations.get('hold_signals', 0):,}")
        
        print("\nFILES CREATED:")
        for file_path in results.get('data_files_created', []):
            print(f"  Data: {file_path}")
        
        for file_path in results.get('visualization_files', []):
            print(f"  Visualization: {file_path}")
        
        print("="*60)

async def main():
    """Main execution function."""
    print("Twitter Market Analysis Pipeline")
    print("===============================")
    print(f"Target: {TARGET_TWEETS} tweets from hashtags: {', '.join(SEARCH_HASHTAGS)}")
    print(f"Time window: Last {TIME_WINDOW_HOURS} hours")
    print()
    
    # Initialize and run pipeline
    pipeline = MarketAnalysisPipeline()
    results = await pipeline.run_complete_analysis()
    
    # Return exit code based on success
    if results['status'] == 'completed':
        print("\n✅ Analysis completed successfully!")
        return 0
    else:
        print(f"\n❌ Analysis failed: {results.get('errors', ['Unknown error'])}")
        return 1

if __name__ == "__main__":
    import sys
    
    try:
        # Run the async main function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)