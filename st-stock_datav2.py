import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

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
# Function to fetch stock performance data
def fetch_stock_performance(tickers, start_date, end_date):
    # Fetch the historical close prices and volumes for the tickers
    data = yf.download(tickers, start=start_date, end=end_date)
    return data
def scrape_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get("marketCap")
        return market_cap
    except Exception as e:
        print(f"Error retrieving market cap for {ticker}: {e}")
        return None  # or an appropriate default value

    


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
    # Plotting the interactive line chart
    st.line_chart(data['Adj Close'])
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
    figsize_width = 26
    figsize_height = num_subplots * 4  # Height of the entire figure
    
    fig, axs = plt.subplots(num_subplots, 5, figsize=(figsize_width, figsize_height))
    
    for i, ticker in enumerate(tickers):
        market_caps = {ticker: scrape_market_cap(ticker) for ticker in tickers}
        max_market_cap = max(market_caps.values())
    
        stock_data = scrape_stock_data(ticker)
        profit_margin = stock_data["Profit Margin"] * 100
        roa = stock_data["ROA"] * 100
        roe = stock_data["ROE"] * 100
    
        # Ticker Labels (First Column)
        axs[i, 0].axis('off')
        axs[i, 0].text(0.5, 0.5, ticker, ha='center', va='center', fontsize=20)
    
        # Market Cap Visualization (Second Column)
        ax1 = axs[i, 1]
        market_cap = market_caps.get(ticker, 0)
        relative_size = market_cap / max_market_cap if max_market_cap > 0 else 0
        ax1.bar(['Market Cap'], [market_cap], color='lightblue')
        ax1.text(0, market_cap, f"{market_cap / 1e9:.2f}B\n({relative_size * 100:.2f}%)", 
                 ha='center', va='bottom', fontsize=16)
        ax1.set_xticks([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
    
        # Financial Metrics (Third Column)
        ax2 = axs[i, 2]
        metrics = [profit_margin, roa, roe]
        metric_names = ["Profit Margin", "ROA", "ROE"]
        ax2.barh(metric_names, metrics, color=['#A3C5A8', '#B8D4B0', '#C8DFBB'])
        for index, value in enumerate(metrics):
            ax2.text(value, index, f" {value:.2f}%", va='center', ha='right' if value < 0 else 'left', fontsize=16)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
        # Revenue Comparison (Fourth Column)
        ax3 = axs[i, 3]
        financials = get_financials(ticker)
        current_year_revenue = financials.loc["Total Revenue"][0]
        previous_year_revenue = financials.loc["Total Revenue"][1]
        revenue_growth = ((current_year_revenue - previous_year_revenue) / previous_year_revenue) * 100
        ax3.bar(["Previous", "Current"], [previous_year_revenue, current_year_revenue], color=['blue', 'orange'])
        ax3.set_xticks([])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        for bar in ax3.patches:
            ax3.annotate(f'{bar.get_height() / 1e9:.2f}B', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='center', color='white', fontsize=14, xytext=(0, 5), textcoords='offset points')
        ax3.annotate(f"Growth: {revenue_growth:.2f}%", xy=(0.5, max(previous_year_revenue, current_year_revenue)), 
                     ha='center', va='bottom', fontsize=16)
    
        # 52-Week Range (Fifth Column)
        ax4 = axs[i, 4]
        week_low = stock_data["52W Low"]
        week_high = stock_data["52W High"]
        current_price = stock_data["Current Price"]
        ax4.plot(["52W Low", "52W High"], [week_low, week_high], color='black', marker='o')
        ax4.scatter(["Current Price"], [current_price], color='red')
        ax4.annotate(f"${current_price:.2f}", xy=("Current Price", current_price), 
                     xytext=(0, 10), textcoords='offset points', ha='center', va='bottom', fontsize=16)
        ax4.set_yticks([])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
