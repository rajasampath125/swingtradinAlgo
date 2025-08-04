import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedPerformanceAnalyzer:
    def __init__(self, trades_data, portfolio_history, initial_capital=10000):
        """
        Advanced performance analysis for LSTM trading strategy
        
        Parameters:
        - trades_data: List of trade dictionaries from your backtesting
        - portfolio_history: List of portfolio value over time
        - initial_capital: Starting capital
        """
        self.trades = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()
        self.portfolio = pd.DataFrame(portfolio_history) if portfolio_history else pd.DataFrame()
        self.initial_capital = initial_capital
        
        print(f"🔬 Advanced Performance Analyzer initialized")
        print(f"📊 Analyzing {len(self.trades)} trades over {len(self.portfolio)} time periods")
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio for risk-adjusted returns"""
        if self.portfolio.empty:
            return None
        
        # Calculate daily returns
        self.portfolio['returns'] = self.portfolio['portfolio_value'].pct_change()
        daily_returns = self.portfolio['returns'].dropna()
        
        # Annualized metrics
        avg_daily_return = daily_returns.mean()
        daily_volatility = daily_returns.std()
        
        # Convert to annual (assuming 4H candles: 6 candles per day * 365 days)
        annual_return = avg_daily_return * (6 * 365)
        annual_volatility = daily_volatility * np.sqrt(6 * 365)
        
        # Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'annual_return': annual_return * 100,
            'annual_volatility': annual_volatility * 100
        }
    
    def analyze_monthly_performance(self):
        """Break down performance by month"""
        if self.trades.empty:
            return None
        
        self.trades['entry_time'] = pd.to_datetime(self.trades['entry_time'])
        self.trades['month_year'] = self.trades['entry_time'].dt.to_period('M')
        
        monthly_stats = self.trades.groupby('month_year').agg({
            'pnl_usd': ['sum', 'count'],
            'pnl_pct': 'mean'
        }).round(2)
        
        monthly_stats.columns = ['Total_PnL_USD', 'Num_Trades', 'Avg_Return_Pct']
        
        # Calculate win rate by month
        monthly_win_rate = self.trades.groupby('month_year').apply(
            lambda x: (x['pnl_pct'] > 0).sum() / len(x) * 100
        ).round(1)
        
        monthly_stats['Win_Rate_Pct'] = monthly_win_rate
        
        return monthly_stats
    
    def analyze_drawdown_periods(self):
        """Detailed drawdown analysis"""
        if self.portfolio.empty:
            return None
        
        portfolio_values = self.portfolio['portfolio_value']
        
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max * 100
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < -1 and not in_drawdown:  # Start of drawdown (>1% loss)
                in_drawdown = True
                start_idx = i
            elif dd >= -0.1 and in_drawdown:  # End of drawdown (recovery)
                in_drawdown = False
                if start_idx is not None:
                    period_dd = drawdown[start_idx:i+1]
                    drawdown_periods.append({
                        'start_date': self.portfolio.iloc[start_idx]['timestamp'],
                        'end_date': self.portfolio.iloc[i]['timestamp'],
                        'duration_periods': i - start_idx + 1,
                        'max_drawdown_pct': period_dd.min(),
                        'recovery_time': i - start_idx + 1
                    })
        
        return pd.DataFrame(drawdown_periods)
    
    def analyze_trade_patterns(self):
        """Analyze patterns in winning and losing trades"""
        if self.trades.empty:
            return None
        
        winners = self.trades[self.trades['pnl_pct'] > 0]
        losers = self.trades[self.trades['pnl_pct'] <= 0]
        
        patterns = {
            'winning_trades': {
                'count': len(winners),
                'avg_return': winners['pnl_pct'].mean(),
                'avg_hold_time': winners['hold_periods'].mean() * 4,  # Convert to hours
                'exit_reasons': winners['exit_reason'].value_counts().to_dict()
            },
            'losing_trades': {
                'count': len(losers),
                'avg_return': losers['pnl_pct'].mean(),
                'avg_hold_time': losers['hold_periods'].mean() * 4,  # Convert to hours
                'exit_reasons': losers['exit_reason'].value_counts().to_dict()
            }
        }
        
        return patterns
    
    def optimization_suggestions(self):
        """Generate optimization suggestions based on analysis"""
        suggestions = []
        
        if not self.trades.empty:
            # Analyze exit reasons
            exit_counts = self.trades['exit_reason'].value_counts()
            
            if exit_counts.get('TIME_EXIT', 0) > exit_counts.get('TAKE_PROFIT', 0):
                suggestions.append("Consider extending take profit levels - many trades hit time limits rather than profit targets")
            
            if exit_counts.get('STOP_LOSS', 0) > len(self.trades) * 0.4:
                suggestions.append("High stop loss rate detected - consider tightening entry criteria or adjusting stop loss levels")
            
            # Win rate analysis
            win_rate = (self.trades['pnl_pct'] > 0).mean() * 100
            if win_rate < 45:
                suggestions.append("Win rate below 45% - focus on improving entry signal accuracy")
            elif win_rate > 65:
                suggestions.append("High win rate detected - you might be able to increase position sizes or take profit targets")
            
            # Return distribution
            avg_win = self.trades[self.trades['pnl_pct'] > 0]['pnl_pct'].mean()
            avg_loss = abs(self.trades[self.trades['pnl_pct'] <= 0]['pnl_pct'].mean())
            
            if avg_win < avg_loss:
                suggestions.append("Average losses exceed average gains - consider adjusting risk/reward ratios")
        
        return suggestions
    
    def generate_comprehensive_report(self):
        """Generate complete performance analysis report"""
        print("\n🔬 ADVANCED PERFORMANCE ANALYSIS REPORT")
        print("=" * 60)
        
        # Sharpe ratio analysis
        sharpe_data = self.calculate_sharpe_ratio()
        if sharpe_data:
            print(f"\n📊 Risk-Adjusted Performance:")
            print(f"Sharpe Ratio: {sharpe_data['sharpe_ratio']:.2f}")
            print(f"Annualized Return: {sharpe_data['annual_return']:.2f}%")
            print(f"Annualized Volatility: {sharpe_data['annual_volatility']:.2f}%")
        
        # Monthly performance
        monthly_perf = self.analyze_monthly_performance()
        if monthly_perf is not None:
            print(f"\n📅 Monthly Performance Summary:")
            print(f"Best Month: {monthly_perf['Total_PnL_USD'].max():.2f} USD")
            print(f"Worst Month: {monthly_perf['Total_PnL_USD'].min():.2f} USD")
            print(f"Average Monthly Trades: {monthly_perf['Num_Trades'].mean():.1f}")
            print(f"Most Active Month: {monthly_perf['Num_Trades'].max()} trades")
        
        # Drawdown analysis
        drawdown_periods = self.analyze_drawdown_periods()
        if drawdown_periods is not None and len(drawdown_periods) > 0:
            print(f"\n📉 Drawdown Analysis:")
            print(f"Number of Drawdown Periods: {len(drawdown_periods)}")
            print(f"Average Drawdown Duration: {drawdown_periods['duration_periods'].mean():.1f} periods")
            print(f"Longest Drawdown: {drawdown_periods['duration_periods'].max()} periods")
        
        # Trade pattern analysis
        patterns = self.analyze_trade_patterns()
        if patterns:
            print(f"\n🎯 Trade Pattern Analysis:")
            print(f"Winning Trades: {patterns['winning_trades']['count']}")
            print(f"  - Avg Return: {patterns['winning_trades']['avg_return']:.2f}%")
            print(f"  - Avg Hold Time: {patterns['winning_trades']['avg_hold_time']:.1f} hours")
            
            print(f"Losing Trades: {patterns['losing_trades']['count']}")
            print(f"  - Avg Return: {patterns['losing_trades']['avg_return']:.2f}%")
            print(f"  - Avg Hold Time: {patterns['losing_trades']['avg_hold_time']:.1f} hours")
        
        # Optimization suggestions
        suggestions = self.optimization_suggestions()
        if suggestions:
            print(f"\n💡 Optimization Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        
        print(f"\n✅ Advanced Analysis Complete!")
        return {
            'sharpe_data': sharpe_data,
            'monthly_performance': monthly_perf,
            'drawdown_periods': drawdown_periods,
            'trade_patterns': patterns,
            'optimization_suggestions': suggestions
        }

# Example usage - you'll need to modify this to use your actual data
def main():
    """Main function to run advanced analysis"""
    print("🎯 LSTM Advanced Performance Analysis - Step 3")
    print("=" * 60)
    
    # You'll need to pass your actual trades and portfolio history here
    # trades_data = your_strategy.trades
    # portfolio_history = your_strategy.portfolio_history
    
    # analyzer = AdvancedPerformanceAnalyzer(trades_data, portfolio_history)
    # results = analyzer.generate_comprehensive_report()
    
    print("📊 To run this analysis, you need to save your trades and portfolio data")
    print("from the trading_strategy.py output and feed them into this analyzer.")

if __name__ == "__main__":
    main()
