import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedPerformanceAnalyzer:
    def __init__(self, trades_data, portfolio_history, initial_capital=10000):
        """Advanced performance analysis for LSTM trading strategy"""
        # Convert to DataFrame and fix timezone issues
        self.trades = self._fix_datetime_columns(pd.DataFrame(trades_data)) if trades_data else pd.DataFrame()
        self.portfolio = self._fix_datetime_columns(pd.DataFrame(portfolio_history)) if portfolio_history else pd.DataFrame()
        self.initial_capital = initial_capital
        
        print(f"🔬 Advanced Performance Analyzer initialized")
        print(f"📊 Analyzing {len(self.trades)} trades over {len(self.portfolio)} time periods")
    
    def _fix_datetime_columns(self, df):
        """Fix timezone-aware datetime columns to prevent conversion errors"""
        df_copy = df.copy()
        
        # Fix common datetime columns
        datetime_cols = ['timestamp', 'entry_time', 'exit_time', 'datetime']
        
        for col in datetime_cols:
            if col in df_copy.columns:
                # Convert to datetime with UTC timezone handling
                df_copy[col] = pd.to_datetime(df_copy[col], utc=True, errors='coerce')
                # Convert to timezone-naive for compatibility
                df_copy[col] = df_copy[col].dt.tz_localize(None)
        
        return df_copy
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio for risk-adjusted returns"""
        if self.portfolio.empty:
            return None
        
        # Calculate period returns
        self.portfolio['returns'] = self.portfolio['portfolio_value'].pct_change()
        period_returns = self.portfolio['returns'].dropna()
        
        if len(period_returns) == 0:
            return None
        
        # Calculate annualized metrics (assuming 4H candles: 6 per day * 365 days)
        periods_per_year = 6 * 365
        avg_period_return = period_returns.mean()
        period_volatility = period_returns.std()
        
        # Annualized metrics
        annual_return = (1 + avg_period_return) ** periods_per_year - 1
        annual_volatility = period_volatility * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'annual_return': annual_return * 100,
            'annual_volatility': annual_volatility * 100,
            'total_periods': len(period_returns)
        }
    
    def analyze_monthly_performance(self):
        """Break down performance by month"""
        if self.trades.empty:
            return None
        
        # Ensure entry_time is datetime
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
        
        # Calculate drawdown percentage
        drawdown = (portfolio_values - running_max) / running_max * 100
        
        # Find significant drawdown periods (>2% loss)
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < -2 and not in_drawdown:  # Start of significant drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= -0.5 and in_drawdown:  # Recovery from drawdown
                in_drawdown = False
                if start_idx is not None:
                    period_dd = drawdown[start_idx:i+1]
                    drawdown_periods.append({
                        'start_date': self.portfolio.iloc[start_idx]['timestamp'],
                        'end_date': self.portfolio.iloc[i]['timestamp'],
                        'duration_periods': i - start_idx + 1,
                        'max_drawdown_pct': period_dd.min(),
                        'recovery_periods': i - start_idx + 1
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
                'exit_reasons': winners['exit_reason'].value_counts().to_dict() if not winners.empty else {},
                'best_trade': winners['pnl_pct'].max() if not winners.empty else 0,
                'median_return': winners['pnl_pct'].median() if not winners.empty else 0
            },
            'losing_trades': {
                'count': len(losers),
                'avg_return': losers['pnl_pct'].mean() if not losers.empty else 0,
                'avg_hold_time': losers['hold_periods'].mean() * 4 if not losers.empty else 0,
                'exit_reasons': losers['exit_reason'].value_counts().to_dict() if not losers.empty else {},
                'worst_trade': losers['pnl_pct'].min() if not losers.empty else 0,
                'median_return': losers['pnl_pct'].median() if not losers.empty else 0
            }
        }
        
        return patterns
    
    def optimization_suggestions(self):
        """Generate optimization suggestions based on analysis"""
        suggestions = []
        
        if not self.trades.empty:
            # Analyze exit reasons
            exit_counts = self.trades['exit_reason'].value_counts()
            total_trades = len(self.trades)
            
            time_exit_pct = (exit_counts.get('TIME_EXIT', 0) / total_trades) * 100
            stop_loss_pct = (exit_counts.get('STOP_LOSS', 0) / total_trades) * 100
            take_profit_pct = (exit_counts.get('TAKE_PROFIT', 0) / total_trades) * 100
            
            if time_exit_pct > 40:
                suggestions.append(f"HIGH TIME EXITS ({time_exit_pct:.1f}%): Consider extending take profit targets or reducing max hold time")
            
            if stop_loss_pct > 40:
                suggestions.append(f"HIGH STOP LOSSES ({stop_loss_pct:.1f}%): Consider tightening entry criteria or adjusting stop loss levels")
            
            if take_profit_pct < 20:
                suggestions.append(f"LOW TAKE PROFITS ({take_profit_pct:.1f}%): Consider more aggressive profit targets")
            
            # Win rate analysis
            win_rate = (self.trades['pnl_pct'] > 0).mean() * 100
            if win_rate < 45:
                suggestions.append(f"Win rate ({win_rate:.1f}%) below optimal - focus on improving entry signal accuracy")
            elif win_rate > 65:
                suggestions.append(f"High win rate ({win_rate:.1f}%) - consider increasing position sizes or profit targets")
            
            # Return distribution analysis
            winners = self.trades[self.trades['pnl_pct'] > 0]
            losers = self.trades[self.trades['pnl_pct'] <= 0]
            
            if not winners.empty and not losers.empty:
                avg_win = winners['pnl_pct'].mean()
                avg_loss = abs(losers['pnl_pct'].mean())
                
                if avg_win < avg_loss:
                    suggestions.append(f"Average losses ({avg_loss:.2f}%) exceed average gains ({avg_win:.2f}%) - adjust risk/reward ratios")
                
                # Risk-reward ratio
                risk_reward = avg_win / avg_loss if avg_loss > 0 else float('inf')
                if risk_reward < 1.2:
                    suggestions.append(f"Risk-reward ratio ({risk_reward:.2f}) is suboptimal - target >1.5")
        
        return suggestions
    
    def generate_comprehensive_report(self):
        """Generate complete performance analysis report"""
        print("\n🔬 ADVANCED PERFORMANCE ANALYSIS REPORT")
        print("=" * 60)
        
        # Basic performance recap
        if not self.trades.empty:
            final_value = self.portfolio['portfolio_value'].iloc[-1] if not self.portfolio.empty else 0
            total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
            
            print(f"\n📊 Performance Summary:")
            print(f"Initial Capital: ${self.initial_capital:,}")
            print(f"Final Value: ${final_value:,.2f}")
            print(f"Total Return: {total_return:+.2f}%")
        
        # Sharpe ratio analysis
        sharpe_data = self.calculate_sharpe_ratio()
        if sharpe_data:
            print(f"\n📈 Risk-Adjusted Performance:")
            print(f"Sharpe Ratio: {sharpe_data['sharpe_ratio']:.3f}")
            print(f"Annualized Return: {sharpe_data['annual_return']:.2f}%")
            print(f"Annualized Volatility: {sharpe_data['annual_volatility']:.2f}%")
            
            # Sharpe ratio interpretation
            if sharpe_data['sharpe_ratio'] > 2.0:
                interpretation = "EXCELLENT (>2.0)"
            elif sharpe_data['sharpe_ratio'] > 1.0:
                interpretation = "GOOD (1.0-2.0)"
            elif sharpe_data['sharpe_ratio'] > 0.5:
                interpretation = "ACCEPTABLE (0.5-1.0)"
            else:
                interpretation = "POOR (<0.5)"
            print(f"Sharpe Rating: {interpretation}")
        
        # Monthly performance
        try:
            monthly_perf = self.analyze_monthly_performance()
            if monthly_perf is not None and len(monthly_perf) > 0:
                print(f"\n📅 Monthly Performance Analysis:")
                print(f"Best Month P&L: ${monthly_perf['Total_PnL_USD'].max():.2f}")
                print(f"Worst Month P&L: ${monthly_perf['Total_PnL_USD'].min():.2f}")
                print(f"Average Monthly Trades: {monthly_perf['Num_Trades'].mean():.1f}")
                print(f"Most Active Month: {monthly_perf['Num_Trades'].max()} trades")
                print(f"Highest Monthly Win Rate: {monthly_perf['Win_Rate_Pct'].max():.1f}%")
                print(f"Lowest Monthly Win Rate: {monthly_perf['Win_Rate_Pct'].min():.1f}%")
        except Exception as e:
            print(f"\n📅 Monthly Performance Analysis: Error - {str(e)}")
        
        # Drawdown analysis
        try:
            drawdown_periods = self.analyze_drawdown_periods()
            if drawdown_periods is not None and len(drawdown_periods) > 0:
                print(f"\n📉 Drawdown Analysis:")
                print(f"Number of Significant Drawdowns (>2%): {len(drawdown_periods)}")
                print(f"Average Drawdown Duration: {drawdown_periods['duration_periods'].mean():.1f} periods")
                print(f"Longest Drawdown: {drawdown_periods['duration_periods'].max()} periods")
                print(f"Deepest Drawdown: {drawdown_periods['max_drawdown_pct'].min():.2f}%")
            else:
                print(f"\n📉 Drawdown Analysis: No significant drawdowns detected (>2%)")
        except Exception as e:
            print(f"\n📉 Drawdown Analysis: Error - {str(e)}")
        
        # Trade pattern analysis
        patterns = self.analyze_trade_patterns()
        if patterns:
            print(f"\n🎯 Detailed Trade Analysis:")
            
            win_data = patterns['winning_trades']
            loss_data = patterns['losing_trades']
            
            print(f"\nWinning Trades ({win_data['count']}):")
            print(f"  - Average Return: {win_data['avg_return']:.2f}%")
            print(f"  - Median Return: {win_data['median_return']:.2f}%")
            print(f"  - Best Trade: {win_data['best_trade']:.2f}%")
            print(f"  - Average Hold: {win_data['avg_hold_time']:.1f} hours")
            
            print(f"\nLosing Trades ({loss_data['count']}):")
            print(f"  - Average Loss: {loss_data['avg_return']:.2f}%")
            print(f"  - Median Loss: {loss_data['median_return']:.2f}%")
            print(f"  - Worst Trade: {loss_data['worst_trade']:.2f}%")
            print(f"  - Average Hold: {loss_data['avg_hold_time']:.1f} hours")
            
            # Risk-reward analysis
            if win_data['avg_return'] > 0 and loss_data['avg_return'] < 0:
                risk_reward = win_data['avg_return'] / abs(loss_data['avg_return'])
                print(f"\nRisk-Reward Ratio: {risk_reward:.2f}")
        
        # Optimization suggestions
        suggestions = self.optimization_suggestions()
        if suggestions:
            print(f"\n💡 Strategy Optimization Recommendations:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        
        print(f"\n✅ Advanced Analysis Complete!")
        print(f"🚀 Ready for Substep 3.2: Strategy Parameter Optimization")
        
        return {
            'sharpe_data': sharpe_data,
            'monthly_performance': monthly_perf if 'monthly_perf' in locals() else None,
            'drawdown_periods': drawdown_periods if 'drawdown_periods' in locals() else None,
            'trade_patterns': patterns,
            'optimization_suggestions': suggestions
        }

def main():
    """Main function to run Substep 3.1: Advanced Analysis"""
    print("🎯 SUBSTEP 3.1: Advanced Performance Analysis (TIMEZONE-FIXED VERSION)")
    print("=" * 60)
    
    try:
        # Load your saved trading data
        with open('sequences/trades_data.pkl', 'rb') as f:
            trades_data = pickle.load(f)
        
        with open('sequences/portfolio_history.pkl', 'rb') as f:
            portfolio_history = pickle.load(f)
        
        print(f"✅ Successfully loaded trading data")
        print(f"📊 {len(trades_data)} trades and {len(portfolio_history)} portfolio points")
        
        # Run advanced analysis with timezone fixes
        analyzer = AdvancedPerformanceAnalyzer(trades_data, portfolio_history, initial_capital=10000)
        results = analyzer.generate_comprehensive_report()
        
        return results
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find saved data files")
        print(f"Make sure trades_data.pkl and portfolio_history.pkl exist in sequences/ folder")
        print(f"Error details: {e}")
        return None
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

if __name__ == "__main__":
    results = main()
