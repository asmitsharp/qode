# Twitter Stock Market Intelligence Scraper

A production-ready Twitter scraping and analysis system for Indian stock market intelligence, designed to collect tweets from key financial hashtags and convert them into quantitative trading signals.

## 🎯 Project Overview

This project addresses the assignment requirements from **Qode Advisors LLP** to build a comprehensive data collection and analysis system for real-time market intelligence. The system efficiently scrapes Twitter for market-related discussions and transforms textual data into actionable trading insights.

### Key Features

- **Modern Twitter Scraping**: Advanced twikit-based approach for reliable data collection
- **Smart Authentication**: Secure cookie-based authentication with environment variable management
- **Real-time Processing**: Efficient data structures optimized for large-scale processing
- **Advanced NLP**: TF-IDF vectorization and sentiment analysis for signal generation
- **Memory-Efficient Visualizations**: Optimized plotting for large datasets
- **Production-Ready**: Comprehensive error handling, logging, and retry mechanisms

## 📊 Technical Implementation

### Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Scraper   │───▶│  Data Processor  │───▶│ Signal Generator│
│   (Multi-method)│    │  (Clean & Store) │    │   (TF-IDF/NLP)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   Visualizer    │◀───│   Main Pipeline  │◀────────────┘
│ (Memory-Efficient)│    │  (Orchestration) │
└─────────────────┘    └──────────────────┘
```

### Core Components

1. **TwitterScraper** (`scraper.py`)

   - Modern twikit-based scraping with guest authentication
   - Intelligent rate limiting and pagination support
   - Robust error handling and retry mechanisms
   - Concurrent hashtag processing with deduplication

2. **DataProcessor** (`data_processor.py`)

   - Data cleaning and normalization
   - Parquet storage with compression
   - Deduplication using content hashing
   - Quality scoring and bot detection

3. **SignalGenerator** (`signal_generator.py`)

   - TF-IDF vectorization for text analysis
   - Multi-dimensional sentiment scoring
   - Composite signal generation with confidence intervals
   - Time-weighted and engagement-weighted signals

4. **MemoryEfficientVisualizer** (`visualizer.py`)
   - Intelligent data sampling for large datasets
   - Interactive and static visualizations
   - Word cloud generation
   - Correlation analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Twitter account credentials (for authentication)
- Internet connection for data collection

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd qode
```

2. **Set up virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
   Create a `.env` file in the project root with your Twitter credentials:

```bash
TWITTER_EMAIL=your_email@example.com
TWITTER_USERNAME=your_username
TWITTER_PASSWORD=your_password
```

5. **Authenticate with Twitter:**

```bash
python auth.py
```

This will create `cookies.json` for authenticated scraping.

### Basic Usage

**Run the complete analysis pipeline:**

```bash
python main.py
```

This will:

- Scrape 2000 tweets from #nifty50, #sensex, #intraday, #banknifty
- Process and clean the data
- Generate trading signals using TF-IDF analysis
- Create comprehensive visualizations
- Save results in Parquet format

### Configuration

we can modify `config.py` to customize:

```python
# Scraping targets
SEARCH_HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty"]
TARGET_TWEETS = 2000
TIME_WINDOW_HOURS = 24

# Rate limiting
MIN_DELAY = 2
MAX_DELAY = 5
MAX_CONCURRENT_REQUESTS = 3

# TF-IDF parameters
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)
```

## 📈 Data Processing Pipeline

### 1. Data Collection

- **Modern twikit scraping**: Reliable API-based data collection with guest/user authentication
- **Target metrics**: username, timestamp, content, likes, retweets, replies, mentions, hashtags
- **Rate limiting**: Intelligent delays and pagination to respect API limits
- **Deduplication**: Content-based hashing to prevent duplicates

### 2. Data Cleaning & Quality Assessment

- **Text normalization**: URL removal, mention cleaning, hashtag processing
- **Quality scoring**: Multi-factor algorithm considering engagement, content length, user patterns
- **Bot detection**: Heuristic-based identification of automated accounts
- **Market relevance**: Keyword-based filtering for financial content

### 3. Signal Generation

- **TF-IDF Analysis**: Convert text to numerical vectors (1000 features, 1-2 grams)
- **Sentiment Scoring**: Bullish/bearish classification using market-specific keywords
- **Temporal Weighting**: Recent tweets get higher importance
- **Engagement Weighting**: Popular tweets influence signals more
- **Confidence Intervals**: Statistical significance measures for signals

### 4. Storage & Optimization

- **Parquet Format**: Efficient columnar storage with Snappy compression
- **Data Types Optimization**: Automatic downcasting for memory efficiency
- **Incremental Updates**: Merge new data with existing datasets
- **Schema Evolution**: Forward-compatible data structures

## 🔧 Advanced Features

### Authentication & Rate Limiting

- **Cookie-based Authentication**: Secure session management with twikit
- **Guest Mode Fallback**: Automatic fallback to guest authentication
- **Intelligent Delays**: Dynamic timing based on market hours
- **Pagination Support**: Efficient data collection with cursor-based pagination
- **Request Pattern Variation**: Human-like request patterns

### Performance Optimizations

- **Memory Management**: Efficient data sampling for visualizations
- **Concurrent Processing**: Async scraping with semaphore limiting
- **Data Compression**: Optimized Parquet storage
- **Garbage Collection**: Automatic memory cleanup
- **Batch Processing**: Chunked data processing for scalability

### Trading Signal Features

- **Multi-dimensional Analysis**: 6+ different signal types
- **Composite Scoring**: Weighted combination of individual signals
- **Time Series Analysis**: Hourly trend aggregation
- **Influence Ranking**: Identification of market-moving tweets
- **Confidence Metrics**: Statistical validation of signals

## 📊 Output Files

### Data Files (stored in `/data/`)

- `raw_tweets.parquet`: Cleaned and processed tweet data
- `trading_signals.parquet`: Generated signals with confidence scores
- `analysis_report.json`: Comprehensive analysis summary

### Visualizations (stored in `/visualizations/`)

- `sentiment_timeline.html`: Interactive time series analysis
- `signal_distribution.png`: Signal strength distributions
- `engagement_analysis.png`: Engagement pattern analysis
- `wordcloud.png`: Market discussion themes
- `correlation_heatmap.png`: Metric correlations
- `interactive_dashboard.html`: Comprehensive dashboard

## 🎯 Signal Interpretation

### Trading Signals

- **BUY**: `primary_signal > 0.6` with `confidence > 0.5`
- **SELL**: `primary_signal < -0.6` with `confidence > 0.5`
- **HOLD**: All other conditions

### Signal Components

- **Primary Signal** (-1 to 1): Composite bullish/bearish indicator
- **Confidence Score** (0 to 1): Statistical reliability measure
- **Signal Strength**: Absolute magnitude of primary signal
- **Time Decay**: Recent tweets weighted higher
- **Engagement Weight**: Popular tweets have more influence

### Current System Limitations

#### HOLD Signal Analysis

Our current system generates a significant number of "HOLD" signals, which indicates neutral market sentiment where the system doesn't have enough confidence to recommend either buying or selling.

**How HOLD Signals Work:**

```python
# Default assignment
df_final['trading_signal'] = 'HOLD'

# BUY only when:
(primary_signal > 0.6) AND (confidence_score > 0.5)

# SELL only when:
(primary_signal < -0.6) AND (confidence_score > 0.5)
```

**The Problem: Overly Conservative Thresholds**
Our current system has extremely strict criteria:

- **Signal Strength**: Requires 60% conviction (0.6 threshold)
- **Confidence**: Requires 50% confidence minimum
- **Low Engagement**: Average engagement rate of only 1.2%

**Why This Happens:**

1. **High Thresholds**: The 0.6 signal strength requirement is very conservative
2. **Engagement Filtering**: Low engagement tweets may not reflect true market sentiment
3. **Time Window**: 24-hour window might miss significant market events

**Potential Improvements:**

- Lower signal thresholds (e.g., 0.4 instead of 0.6)
- Increase data collection volume
- Implement sentiment momentum analysis
- Add market volatility indicators
- Include more diverse hashtags and sources

## 🔍 Technical Deep Dive

### Approach & Methodology

#### 1. **Modern Twikit Scraping Strategy**

- **Primary Method**: Twikit library for reliable API-based scraping
- **Guest Authentication**: Automatic fallback to guest mode when needed
- **Cookie Management**: Persistent session management for efficiency
- **Pagination Support**: Cursor-based pagination for comprehensive data collection

#### 2. **NLP Signal Generation**

```python
# TF-IDF Vectorization
tfidf_matrix = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
).fit_transform(texts)

# Dimensionality Reduction
reduced_features = TruncatedSVD(n_components=50).fit_transform(tfidf_matrix)

# Signal Composition
primary_signal = weighted_sum([
    sentiment_score * 0.3,
    tfidf_momentum * 0.25,
    engagement_weight * 0.2,
    time_decay * 0.15,
    quality_score * 0.1
])
```

#### 3. **Design Patterns Used**

- **Strategy Pattern**: Multiple scraping methods
- **Observer Pattern**: Event-driven processing pipeline
- **Factory Pattern**: Dynamic signal generator creation
- **Singleton Pattern**: Configuration management
- **Template Method**: Consistent visualization generation

#### 4. **Concepts & Techniques**

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Convert text to numerical features
- **Truncated SVD**: Dimensionality reduction for noise reduction
- **K-Means Clustering**: Topic identification and theme analysis
- **Exponential Backoff**: Intelligent retry mechanism
- **Memory Mapping**: Efficient large dataset handling
- **Async Programming**: Non-blocking I/O operations

## 🛡️ Production Considerations

### Error Handling

- **Retry Logic**: Exponential backoff for failed requests
- **Graceful Degradation**: Fallback methods when primary fails
- **Comprehensive Logging**: Detailed execution tracking
- **Exception Isolation**: Individual failures don't crash pipeline

### Scalability Features

- **Memory Optimization**: Efficient data structures and cleanup
- **Concurrent Processing**: Async operations where beneficial
- **Data Streaming**: Process data in chunks to handle large volumes
- **Modular Design**: Easy to scale individual components

### Monitoring & Observability

- **Execution Metrics**: Detailed performance tracking
- **Quality Scores**: Data quality assessment at each stage
- **Signal Validation**: Statistical significance testing
- **Resource Usage**: Memory and time optimization tracking

## 📚 Dependencies

### Core Libraries

- **pandas**: Data manipulation and analysis
- **scikit-learn**: TF-IDF vectorization and clustering
- **beautifulsoup4**: HTML parsing
- **aiohttp**: Async HTTP requests
- **fastparquet**: Efficient data storage

### Visualization

- **matplotlib/seaborn**: Statistical plotting
- **plotly**: Interactive visualizations
- **wordcloud**: Text visualization

### Utilities

- **fake-useragent**: User agent rotation
- **asyncio-throttle**: Rate limiting
- **python-dotenv**: Configuration management

## 🔧 Configuration Options

### Scraping Parameters

```python
TARGET_TWEETS = 2000                    # Total tweets to collect
TIME_WINDOW_HOURS = 24                 # Time window for tweets
MAX_CONCURRENT_REQUESTS = 3            # Concurrent scraping limit
```

### Signal Processing

```python
SIGNAL_THRESHOLD_POSITIVE = 0.6        # BUY signal threshold
SIGNAL_THRESHOLD_NEGATIVE = -0.6       # SELL signal threshold
TFIDF_MAX_FEATURES = 1000              # TF-IDF feature count
```

### Performance Tuning

```python
MIN_DELAY = 2                          # Minimum request delay
MAX_DELAY = 5                          # Maximum request delay
RETRY_ATTEMPTS = 3                     # Retry count for failures
```

## 🎯 Assignment Alignment

This solution addresses all key requirements from the Qode Advisors assignment:

### ✅ **Data Collection**

- Scrapes 500+ tweets from specified hashtags (#nifty50, #sensex, #intraday, #banknifty)
- Extracts all required fields: username, timestamp, content, engagement metrics, mentions, hashtags
- Modern twikit-based approach with guest authentication
- Targets last 24-hour window with temporal filtering

### ✅ **Technical Implementation**

- Efficient data structures with optimized memory usage
- Creative rate limiting and anti-bot circumvention
- Time/space complexity optimization throughout
- Production-ready code with comprehensive error handling and logging

### ✅ **Data Processing & Storage**

- Complete data cleaning and normalization pipeline
- Parquet format storage with compression optimization
- Robust deduplication using content hashing
- Unicode and special character handling for Indian content

### ✅ **Analysis & Insights**

- **Text-to-Signal Conversion**: TF-IDF vectorization with market-specific features
- **Memory-efficient Visualization**: Intelligent data sampling and streaming plots
- **Signal Aggregation**: Multi-factor composite signals with confidence intervals
- Quantitative trading recommendations (BUY/SELL/HOLD)

### ✅ **Performance Optimization**

- Concurrent async processing with semaphore limiting
- Memory-efficient data handling with type optimization
- Designed for 10x scalability with modular architecture
- Garbage collection and resource cleanup

### ✅ **Professional Deliverables**

- Clean GitHub repository structure
- Comprehensive README with setup instructions
- Complete requirements.txt and environment setup
- Sample output data and professional documentation
- Technical documentation explaining approach and methodology

## 🚀 Future Enhancements

1. **Real-time Streaming**: WebSocket integration for live data
2. **Advanced ML**: Deep learning models for sentiment analysis
3. **Multi-language Support**: Hindi and other Indian language processing
4. **API Integration**: RESTful API for external consumption
5. **Cloud Deployment**: Scalable cloud infrastructure setup
6. **Alert System**: Real-time notifications for significant signals
