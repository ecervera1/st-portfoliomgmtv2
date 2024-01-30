import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# Function to scrape summary stock data
def scrape_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    growth_ratio = info.get("revenueGrowth")
    pe_ratio = info.get("trailingPE")
    earnings_growth = info.get("revenueGrowth")

    data = {
            "Current Price": info.get("currentPrice"),
            "Market Cap (B)": info.get("marketCap") / 1e9 if info.get("marketCap") else None, 
            "PE Ratio": info.get("trailingPE"),
            "PEG Ratio": info.get("pegRatio"),
            "Profit Margin": info.get("profitMargins"),
            "ROA": info.get("returnOnAssets"),
            "ROE": info.get("returnOnEquity"),
            "52W Range": f"{info.get('fiftyTwoWeekLow')} - {info.get('fiftyTwoWeekHigh')}",
            "52W Low": info.get("fiftyTwoWeekLow"),
            "52W High":info.get("fiftyTwoWeekHigh"),
            "Div Yield": info.get("dividendYield"),
            "Beta": info.get("beta"),
            "Forward Annual Dividend Yield": info.get("dividendYield") or "-",
            "EPS per Year": info.get("trailingEps"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Earnings Growth": info.get("earningsGrowth")
        }
    return data

# Function to scrape market cap data
def scrape_market_cap(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    market_cap = info.get("marketCap")
    return market_cap

# Function to fetch financial metrics
def fetch_financial_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "Profit Margin": info.get("profitMargins"),
            "ROA": info.get("returnOnAssets"),
            "ROE": info.get("returnOnEquity")
        }
    except Exception as e:
        st.error(f"Error fetching financial metrics for {ticker}: {e}")
        return {}

# Function to fetch stock performance data
def fetch_stock_performance(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching stock performance data: {e}")
        return pd.DataFrame()

# Function to get financials
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        return financials
    except Exception as e:
        st.error(f"Error fetching financials for {ticker}: {e}")
        return pd.DataFrame()

# Streamlit app layout
st.title('Portfolio Management - Stock Comparative Analysis')

# Input for stock tickers
user_input = st.text_input("Enter stock tickers separated by commas", "LLY, ABT, MRNA, JNJ, BIIB, BMY, PFE, AMGN, WBA")

# Input for date range
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-22"))

# Button to run the scraper and plot stock performance
if st.button('Run'):
    # Split the user input into a list of tickers
    tickers = [ticker.strip() for ticker in user_input.split(',')]

    # Plot stock performance
    data = fetch_stock_performance(tickers, start_date, end_date)

    st.title('Stock Performance Chart')
    st.markdown(f'({start_date} - {end_date})')

    # Plotting the interactive line chart with Plotly
    fig = px.line(data, x=data.index, y=data.columns, title='Stock Performance')
    st.plotly_chart(fig)

    st.title('Stock Data')

    # Create an empty list to store dictionaries of stock data
    stock_data_list = []

    # Loop through each ticker, scrape the data, and add it to the list
    for ticker in tickers:
        try:
            ticker_data = scrape_stock_data(ticker)
            stock_data_list.append(ticker_data)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    # Create a DataFrame from the list of dictionaries
    stock_data_df = pd.DataFrame(stock_data_list, index=tickers)

    # Transpose the DataFrame
    stock_data_transposed = stock_data_df.transpose()
    stock_data_transposed.fillna('-', inplace=True)

    for col in stock_data_transposed.columns:
        if col != "52W Range":  # Exclude the "52W Range" column
            stock_data_transposed[col] = stock_data_transposed[col].apply(
                lambda x: f'{x:.2f}' if isinstance(x, float) else x)

    # Display the DataFrame as a table
    st.table(stock_data_transposed)

    # Creating Charts
    num_subplots = len(tickers)

    # Create a list to store Plotly figures
    plotly_figs = []

    for i, ticker in enumerate(tickers):
        # Get market cap data
        market_cap = scrape_market_cap(ticker)
        max_market_cap = max(market_caps.values())

        # Create a Plotly figure for Market Cap
        fig_market_cap = px.pie(names=[ticker], values=[market_cap], title=f'{ticker} Market Cap')
        plotly_figs.append(fig_market_cap)

        # Extract Profit Margin, ROA, and ROE values and convert to percentage
        profit_margin = stock_data["Profit Margin"] * 100
        roa = stock_data["ROA"] * 100 if isinstance(stock_data["ROA"], (float, int)) and stock_data["ROA"] > 0 else 0
        roe = stock_data["ROE"] * 100 if isinstance(stock_data["ROE"], (float, int)) and stock_data["ROE"] > 0 else 0

        # Create a Plotly figure for Financial Metrics
        fig_metrics = px.bar(x=[ticker], y=[profit_margin, roa, roe],
                             labels={'x': 'Metric', 'y': 'Value'},
                             title=f'{ticker} - Financial Metrics',
                             color=['Profit Margin', 'ROA', 'ROE'],
                             color_discrete_map={'Profit Margin': '#A3C5A8', 'ROA': '#B8D4B0', 'ROE': '#C8DFBB'})
        plotly_figs.append(fig_metrics)

        # Create a Plotly figure for Revenue Comparison
        fig_revenue = px.bar(x=['2022', '2023'], y=[previous_year_revenue_billion, current_year_revenue_billion],
                             labels={'x': 'Year', 'y': 'Revenue (Billions)'},
                             title=f'{ticker} Revenue Comparison (2022 vs 2023)',
                             color=['2022', '2023'],
                             color_discrete_map={'2022': 'blue', '2023': 'orange'})
        fig_revenue.update_traces(text=[round(previous_year_revenue_billion, 2), round(current_year_revenue_billion, 2)])
        fig_revenue.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
        plotly_figs.append(fig_revenue)

        # Create a Plotly figure for 52-Week Range
        fig_range = px.bar(x=['52W Low', 'Current Price', '52W High'],
                           y=[week_low, current_price, week_high],
                           labels={'x': 'Range', 'y': 'Price'},
                           title=f'{ticker} 52-Week Range',
                           color=['52W Low', 'Current Price', '52W High'],
                           color_discrete_map={'52W Low': 'black', 'Current Price': 'red', '52W High': 'black'})
        fig_range.update_traces(text=[f'${week_low:.2f}', f'${current_price:.2f}', f'${week_high:.2f}'])
        fig_range.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
        plotly_figs.append(fig_range)

    # Display the Plotly figures
    for fig in plotly_figs:
        st.plotly_chart(fig)
