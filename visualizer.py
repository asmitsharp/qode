import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from typing import List, Dict, Any, Optional, Tuple
import warnings
from datetime import datetime, timezone
import os

from config import *
from utils import setup_logging

warnings.filterwarnings('ignore')

class MemoryEfficientVisualizer:
    
    def __init__(self):
        self.logger = setup_logging(LOG_LEVEL)
        self.ensure_output_directory()
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def ensure_output_directory(self):
        os.makedirs("visualizations", exist_ok=True)
        
    def sample_data_efficiently(self, df: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
        if len(df) <= max_points:
            return df
        
        if 'trading_signal' in df.columns:
            samples = []
            for signal in df['trading_signal'].unique():
                signal_df = df[df['trading_signal'] == signal]
                n_samples = min(len(signal_df), max_points // len(df['trading_signal'].unique()))
                if n_samples > 0:
                    samples.append(signal_df.sample(n=n_samples, random_state=42))
            
            if samples:
                return pd.concat(samples, ignore_index=True)
        
        # Fallback: random sampling
        return df.sample(n=max_points, random_state=42)
    
    def create_sentiment_timeline(self, df: pd.DataFrame, save_path: str = None) -> str:
        if df.empty or 'timestamp' not in df.columns:
            self.logger.warning("Insufficient data for sentiment timeline")
            return ""
        
        df_sample = self.sample_data_efficiently(df, 2000)
        
        df_hourly = df_sample.set_index('timestamp').resample('1H').agg({
            'primary_signal': 'mean',
            'composite_sentiment': 'mean',
            'confidence_score': 'mean',
            'total_engagement': 'sum'
        }).reset_index()
        
        # Remove empty periods
        df_hourly = df_hourly.dropna()
        
        if df_hourly.empty:
            return ""
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Market Sentiment Over Time', 'Signal Confidence', 'Engagement Volume'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_hourly['timestamp'],
                y=df_hourly['primary_signal'],
                mode='lines+markers',
                name='Primary Signal',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        
        fig.add_trace(
            go.Scatter(
                x=df_hourly['timestamp'],
                y=df_hourly['confidence_score'],
                mode='lines',
                name='Confidence',
                line=dict(color='orange', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.3)'
            ),
            row=2, col=1
        )
        
        # Engagement volume
        fig.add_trace(
            go.Bar(
                x=df_hourly['timestamp'],
                y=df_hourly['total_engagement'],
                name='Engagement',
                marker_color='green',
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Market Sentiment Analysis - Time Series",
            showlegend=True,
            template="plotly_white"
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Signal Strength", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        fig.update_yaxes(title_text="Total Engagement", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        # Save plot
        if not save_path:
            save_path = "visualizations/sentiment_timeline.html"
        
        fig.write_html(save_path)
        self.logger.info(f"Sentiment timeline saved to {save_path}")
        
        return save_path
    
    def create_signal_distribution(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create signal strength distribution plot."""
        if df.empty:
            return ""
        
        # Sample data
        df_sample = self.sample_data_efficiently(df, 5000)
        
        fig, axes = plt.subplots(2, 2, figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
        fig.suptitle('Trading Signal Analysis', fontsize=16, fontweight='bold')
        
        # Signal strength distribution
        if 'primary_signal' in df_sample.columns:
            axes[0, 0].hist(df_sample['primary_signal'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_title('Signal Strength Distribution')
            axes[0, 0].set_xlabel('Signal Strength')
            axes[0, 0].set_ylabel('Frequency')
        
        # Confidence vs Signal scatter
        if all(col in df_sample.columns for col in ['primary_signal', 'confidence_score']):
            scatter = axes[0, 1].scatter(
                df_sample['primary_signal'], 
                df_sample['confidence_score'],
                alpha=0.6, 
                c=df_sample['total_engagement'] if 'total_engagement' in df_sample else 'blue',
                cmap='viridis',
                s=30
            )
            axes[0, 1].set_title('Signal vs Confidence')
            axes[0, 1].set_xlabel('Primary Signal')
            axes[0, 1].set_ylabel('Confidence Score')
            if 'total_engagement' in df_sample:
                plt.colorbar(scatter, ax=axes[0, 1], label='Engagement')
        
        # Trading signal pie chart
        if 'trading_signal' in df_sample.columns:
            signal_counts = df_sample['trading_signal'].value_counts()
            axes[1, 0].pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Trading Signal Distribution')
        
        # Sentiment by hour
        if 'timestamp' in df_sample.columns and 'primary_signal' in df_sample.columns:
            df_sample['hour'] = pd.to_datetime(df_sample['timestamp']).dt.hour
            hourly_sentiment = df_sample.groupby('hour')['primary_signal'].mean()
            axes[1, 1].plot(hourly_sentiment.index, hourly_sentiment.values, marker='o')
            axes[1, 1].set_title('Average Sentiment by Hour')
            axes[1, 1].set_xlabel('Hour of Day')
            axes[1, 1].set_ylabel('Average Signal')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if not save_path:
            save_path = "visualizations/signal_distribution.png"
        
        plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_DPI)
        plt.close()  # Free memory
        
        self.logger.info(f"Signal distribution saved to {save_path}")
        return save_path
    
    def create_engagement_analysis(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create engagement analysis visualization."""
        if df.empty:
            return ""
        
        df_sample = self.sample_data_efficiently(df, 3000)
        
        fig, axes = plt.subplots(2, 2, figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
        fig.suptitle('Engagement Analysis', fontsize=16, fontweight='bold')
        
        # Engagement distribution
        if 'total_engagement' in df_sample.columns:
            # Log scale for better visualization
            engagement_log = np.log1p(df_sample['total_engagement'])
            axes[0, 0].hist(engagement_log, bins=30, alpha=0.7, color='lightgreen')
            axes[0, 0].set_title('Engagement Distribution (Log Scale)')
            axes[0, 0].set_xlabel('Log(1 + Total Engagement)')
            axes[0, 0].set_ylabel('Frequency')
        
        # Engagement vs Signal
        if all(col in df_sample.columns for col in ['total_engagement', 'primary_signal']):
            axes[0, 1].scatter(
                np.log1p(df_sample['total_engagement']), 
                df_sample['primary_signal'],
                alpha=0.6, s=30
            )
            axes[0, 1].set_title('Engagement vs Signal Strength')
            axes[0, 1].set_xlabel('Log(1 + Total Engagement)')
            axes[0, 1].set_ylabel('Primary Signal')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Top users by engagement
        if all(col in df_sample.columns for col in ['username', 'total_engagement']):
            top_users = df_sample.groupby('username')['total_engagement'].sum().nlargest(10)
            axes[1, 0].barh(range(len(top_users)), top_users.values)
            axes[1, 0].set_yticks(range(len(top_users)))
            axes[1, 0].set_yticklabels(top_users.index, fontsize=8)
            axes[1, 0].set_title('Top 10 Users by Engagement')
            axes[1, 0].set_xlabel('Total Engagement')
        
        # Engagement types breakdown
        engagement_cols = ['likes', 'retweets', 'replies']
        available_cols = [col for col in engagement_cols if col in df_sample.columns]
        
        if available_cols:
            engagement_sums = [df_sample[col].sum() for col in available_cols]
            axes[1, 1].pie(engagement_sums, labels=available_cols, autopct='%1.1f%%')
            axes[1, 1].set_title('Engagement Types Distribution')
        
        plt.tight_layout()
        
        if not save_path:
            save_path = "visualizations/engagement_analysis.png"
        
        plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_DPI)
        plt.close()
        
        self.logger.info(f"Engagement analysis saved to {save_path}")
        return save_path
    
    def create_wordcloud_visualization(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create word cloud from tweet content."""
        if df.empty or 'content_cleaned' not in df.columns:
            return ""
        
        # Combine all text
        all_text = ' '.join(df['content_cleaned'].dropna().tolist())
        
        if not all_text.strip():
            return ""
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=MAX_WORDS_WORDCLOUD,
            stopwords=STOP_WORDS_CUSTOM,
            collocations=False,
            prefer_horizontal=0.7
        ).generate(all_text)
        
        # Create plot
        plt.figure(figsize=(12, 6), dpi=PLOT_DPI)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Market Discussion Word Cloud', fontsize=16, fontweight='bold', pad=20)
        
        if not save_path:
            save_path = "visualizations/wordcloud.png"
        
        plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_DPI, facecolor='white')
        plt.close()
        
        self.logger.info(f"Word cloud saved to {save_path}")
        return save_path
    
    def create_correlation_heatmap(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create correlation heatmap of key metrics."""
        if df.empty:
            return ""
        
        # Select numeric columns for correlation
        numeric_cols = [
            'primary_signal', 'confidence_score', 'composite_sentiment',
            'bullish_score', 'bearish_score', 'total_engagement',
            'likes', 'retweets', 'replies', 'content_length',
            'hashtag_count', 'mention_count', 'quality_score'
        ]
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 3:
            self.logger.warning("Insufficient numeric columns for correlation analysis")
            return ""
        
        # Sample data for memory efficiency
        df_sample = self.sample_data_efficiently(df, 5000)
        
        # Calculate correlation matrix
        corr_matrix = df_sample[available_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10), dpi=PLOT_DPI)
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('Correlation Matrix of Key Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if not save_path:
            save_path = "visualizations/correlation_heatmap.png"
        
        plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_DPI)
        plt.close()
        
        self.logger.info(f"Correlation heatmap saved to {save_path}")
        return save_path
    
    def create_interactive_dashboard(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create interactive dashboard with key metrics."""
        if df.empty:
            return ""
        
        # Sample data
        df_sample = self.sample_data_efficiently(df, 2000)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Signal Strength Over Time', 
                'Trading Signal Distribution',
                'Top Users by Influence', 
                'Sentiment vs Engagement'
            ),
            specs=[
                [{"secondary_y": True}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # Time series plot
        if all(col in df_sample.columns for col in ['timestamp', 'primary_signal']):
            fig.add_trace(
                go.Scatter(
                    x=df_sample['timestamp'],
                    y=df_sample['primary_signal'],
                    mode='markers',
                    name='Signal Strength',
                    marker=dict(size=6, opacity=0.7)
                ),
                row=1, col=1
            )
        
        # Trading signal pie
        if 'trading_signal' in df_sample.columns:
            signal_counts = df_sample['trading_signal'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=signal_counts.index,
                    values=signal_counts.values,
                    name="Trading Signals"
                ),
                row=1, col=2
            )
        
        # Top users
        if all(col in df_sample.columns for col in ['username', 'influence_score']):
            top_users = df_sample.groupby('username')['influence_score'].mean().nlargest(10)
            fig.add_trace(
                go.Bar(
                    x=top_users.values,
                    y=top_users.index,
                    orientation='h',
                    name='Top Users'
                ),
                row=2, col=1
            )
        
        # Sentiment vs engagement scatter
        if all(col in df_sample.columns for col in ['primary_signal', 'total_engagement']):
            fig.add_trace(
                go.Scatter(
                    x=df_sample['primary_signal'],
                    y=df_sample['total_engagement'],
                    mode='markers',
                    name='Tweets',
                    marker=dict(
                        size=8,
                        opacity=0.6,
                        color=df_sample['confidence_score'] if 'confidence_score' in df_sample else 'blue',
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Twitter Market Analysis Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        # Save interactive plot
        if not save_path:
            save_path = "visualizations/interactive_dashboard.html"
        
        fig.write_html(save_path)
        self.logger.info(f"Interactive dashboard saved to {save_path}")
        
        return save_path
    
    def create_comprehensive_report(self, df: pd.DataFrame, summary_stats: Dict[str, Any]) -> List[str]:
        """Create comprehensive visualization report."""
        if df.empty:
            return []
        
        self.logger.info("Creating comprehensive visualization report")
        
        created_files = []
        
        # Create all visualizations
        visualizations = [
            self.create_sentiment_timeline,
            self.create_signal_distribution,
            self.create_engagement_analysis,
            self.create_wordcloud_visualization,
            self.create_correlation_heatmap,
            self.create_interactive_dashboard
        ]
        
        for viz_func in visualizations:
            try:
                result = viz_func(df)
                if result:
                    created_files.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to create visualization {viz_func.__name__}: {e}")
        
        # Create summary statistics file
        if summary_stats:
            summary_path = "visualizations/summary_statistics.txt"
            with open(summary_path, 'w') as f:
                f.write("=== TWITTER MARKET ANALYSIS SUMMARY ===\n\n")
                f.write(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n")
                
                for section, data in summary_stats.items():
                    f.write(f"{section.upper()}:\n")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {data}\n")
                    f.write("\n")
            
            created_files.append(summary_path)
        
        self.logger.info(f"Created {len(created_files)} visualization files")
        return created_files
    
    def optimize_memory_usage(self):
        """Clear matplotlib and seaborn cache to free memory."""
        plt.close('all')
        import gc
        gc.collect()
        self.logger.info("Memory optimization completed")