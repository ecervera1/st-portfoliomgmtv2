import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from datetime import datetime
import re


custom_css = """
<style>
    .stActionButton button[kind="header"] {
        visibility: hidden;
    }

    .stActionButton div[data-testid="stActionButtonIcon"] {
        visibility: hidden;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)





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
        "52W High": info.get("fiftyTwoWeekHigh"),
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
        
def get_financial_statements(ticker):
    stock = yf.Ticker(ticker)
    financial_statements = {
        "income_statement": stock.financials,
        "balance_sheet": stock.balance_sheet,
        "cash_flow": stock.cashflow
    }
    return financial_statements


# Streamlit app layout
st.title('Portfolio Management - Stock Comparative Analysis')

# Sidebar for user inputs
st.sidebar.title('Input Parameters')

# Input for stock tickers
user_input = st.sidebar.text_input("Enter stock tickers separated by commas", "LLY, ABT, MRNA, JNJ, BIIB, BMY, PFE, AMGN, WBA")
tickers = [ticker.strip() for ticker in user_input.split(',')]

selected_stock = st.sidebar.selectbox("Select a Stock", tickers)

# Input for date range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))

# Set the default "End Date" value to be today's date
default_end_date = datetime.today().date()
end_date = st.sidebar.date_input("End Date", default_end_date)

def get_financial_value(df, pattern, year_offset=0):
    for label in df.index:
        if re.search(pattern, label, re.IGNORECASE):
            if 0 <= year_offset < len(df.columns):
                return df.loc[label].iloc[-(year_offset + 1)]
            break
    return 0.0



# Function to extract the year from a DataFrame column

def extract_year_from_column(column):
    try:
        return pd.to_datetime(column).year
    except Exception as e:
        print(f"Error in extracting year: {e}")
        return None


# Function to calculate FCFF and FCFE
def calculate_fcff_and_fcfe(ticker):
    tickerData = yf.Ticker(ticker)
    income_statement = tickerData.financials
    cash_flow = tickerData.cashflow
    balance_sheet = tickerData.balance_sheet

    results = pd.DataFrame()


    years_in_reverse = list(reversed(income_statement.columns))

    for i, column in enumerate(years_in_reverse):
        column_date = pd.to_datetime(column)  # Convert the column to a datetime object
        year = column_date.year  # Extract the year from the datetime object



        net_income = get_financial_value(income_statement, 'Net Income', i)
        depreciation = get_financial_value(cash_flow, 'Depreciation And Amortization', i)
        interest_expense = get_financial_value(income_statement, 'Interest Expense', i)
        tax_expense = get_financial_value(income_statement, 'Tax Provision', i)
        income_before_tax = get_financial_value(income_statement, 'Pretax Income', i)
        tax_rate = tax_expense / income_before_tax if income_before_tax != 0 else 0.21  # Fallback to a default tax rate
        capex = get_financial_value(cash_flow, 'Capital Expenditure', i)
        net_borrowing = get_financial_value(cash_flow, 'Issuance Of Debt', i) - get_financial_value(cash_flow, 'Repayment Of Debt', i)
        current_assets = get_financial_value(balance_sheet, 'Total Current Assets', i)
        previous_current_assets = get_financial_value(balance_sheet, 'Total Current Assets', i+1)
        current_liabilities = get_financial_value(balance_sheet, 'Total Current Liabilities', i)
        previous_current_liabilities = get_financial_value(balance_sheet, 'Total Current Liabilities', i+1)
        #change_in_nwc = (current_assets - previous_current_assets) - (current_liabilities - previous_current_liabilities)
        change_in_nwc = get_financial_value(cash_flow,'Change In Working Capital', i)

        # Calculate FCFF and FCFE
        fcff = net_income + depreciation + (interest_expense * (1 - tax_rate)) - capex - change_in_nwc
        fcfe = fcff - (interest_expense * (1 - tax_rate)) + net_borrowing

        # Append the calculations to the results DataFrame
        new_row = pd.DataFrame({'Year': [year], 'Net Income': [net_income], 'Depreciation': [depreciation],
                                'Interest Expense': [interest_expense], 'Tax Expense': [tax_expense],
                                'Income Before Tax': [income_before_tax], 'CapEx': [capex],
                                'Net Borrowing': [net_borrowing], 'Change in NWC': [change_in_nwc],
                                'Tax Rate': [tax_rate], 'FCFF': [fcff], 'FCFE': [fcfe]})
        results = pd.concat([results, new_row], ignore_index=True)
        

    return results
    #print(results)


# Button to run the scraper and plot stock performance
if st.sidebar.button('Run'):
    # Split the user input into a list of tickers
    #tickers = [ticker.strip() for ticker in user_input.split(',')]

    # Plot stock performance
    data = fetch_stock_performance(tickers, start_date, end_date)

    st.title('Stock Performance Chart')
    # Format the date range for the selected date range
    formatted_start_date = start_date.strftime("%Y-%m-%d")
    formatted_end_date = end_date.strftime("%Y-%m-%d")

    st.markdown(f'({formatted_start_date} - {formatted_end_date})')
    
    # Plotting the interactive line chart
    st.line_chart(data['Adj Close'])

    last_10_years_end_date = end_date
    last_10_years_start_date = last_10_years_end_date - pd.DateOffset(years=10)
    data_last_10_years = fetch_stock_performance(tickers, last_10_years_start_date, last_10_years_end_date)

    st.title('Stock Performance Chart (Last 10 Years)')
    formatted_last_10_years_start_date = last_10_years_start_date.strftime("%b-%y")
    formatted_last_10_years_end_date = last_10_years_end_date.strftime("%b-%y")

    st.markdown(f'({formatted_last_10_years_start_date} - {formatted_last_10_years_end_date})')

    # Plotting the interactive line chart for the last 10 years
    st.line_chart(data_last_10_years['Adj Close'])
    
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
    num_subplots = len(tickers) + 1
    figsize_width =  28
    figsize_height = num_subplots * 4  # Height of the entire figure

    # Create a figure with subplots: X columns (Ticker, Market Cap, Revenue, Financial Metrics...) for each ticker
    fig, axs = plt.subplots(num_subplots, 5, figsize=(figsize_width, figsize_height), gridspec_kw={'wspace': 0.5})

    # Adding labels in the first row
    labels = ["Ticker", "Market Cap", "Financial Metrics", "Revenue Comparison", "52-Week Range"]
    for j in range(5):
        axs[0, j].axis('off')
        axs[0, j].text(0.5, 0.5, labels[j], ha='center', va='center', fontsize=25, fontweight='bold')

    for i, ticker in enumerate(tickers, start=1):

        # Function to scrape market cap data
        def scrape_market_cap(ticker):
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get("marketCap")
            return market_cap
    
        # Get market cap data
        market_caps = {ticker: scrape_market_cap(ticker) for ticker in tickers}
        
        # Find the largest market cap for scaling
        max_market_cap = max(market_caps.values())
        
        #Scrape data for the ticker
        stock_data = scrape_stock_data(ticker)
        
        # Extract Profit Margin, ROA, and ROE values and convert to percentage
        profit_margin = stock_data["Profit Margin"] * 100
        roa = stock_data["ROA"] * 100 if isinstance(stock_data["ROA"], (float, int)) and stock_data["ROA"] > 0 else 0
        roe = stock_data["ROE"] * 100 if isinstance(stock_data["ROE"], (float, int)) and stock_data["ROE"] > 0 else 0

        # Ticker Labels (First Column)
        axs[i, 0].axis('off')
        axs[i, 0].text(0.5, 0.5, ticker, ha='center', va='center', fontsize=30)

        # Market Cap Visualization (Second Column)
        ax1 = axs[i, 1]
        market_cap = market_caps.get(ticker, 0)
        relative_size = market_cap / max_market_cap if max_market_cap > 0 else 0
        circle = plt.Circle((0.5, 0.5), relative_size * 0.5, color='lightblue')
        ax1.add_artist(circle)
        ax1.set_aspect('equal', adjustable='box')
        text = ax1.text(0.5, 0.5, f"{market_cap / 1e9:.2f}B", ha='center', va='center', fontsize=20)
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Adjust bar width for less padding
        bar_width = 1
        
        # ROE ROA and PM      
        # Financial Metrics (Third Column)
        ax2 = axs[i, 2]
        metrics = [profit_margin, roa, roe]
        metric_names = ["Profit Margin", "ROA", "ROE"]
        bars = ax2.barh(metric_names, metrics, color=['#A3C5A8', '#B8D4B0', '#C8DFBB'])
        
        for index, (label, value) in enumerate(zip(metric_names, metrics)):
            # Adjusting the position dynamically
            label_x_offset = max(-1, -0.1 * len(str(value)))
            ax2.text(label_x_offset, index, label, va='center', ha='right', fontsize=16)
        
            # Add value label
            value_x_position = value + 1 if value >= 0 else value - 1
            ax2.text(value_x_position, index, f"{value:.2f}%", va='center', ha='left' if value >= 0 else 'right', fontsize=16)
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Revenue Comparison (Third Column)
        ax3 = axs[i, 3]
        financials = get_financials(ticker)
        current_year_revenue = financials.loc["Total Revenue"][0]
        previous_year_revenue = financials.loc["Total Revenue"][1]
    
        current_year_revenue_billion = current_year_revenue / 1e9
        previous_year_revenue_billion = previous_year_revenue / 1e9
        growth = ((current_year_revenue_billion - previous_year_revenue_billion) / previous_year_revenue_billion) * 100
    
        line_color = 'green' if growth > 0 else 'red'
    
        bars = ax3.bar(["2022", "2023"], [previous_year_revenue_billion, current_year_revenue_billion], color=['blue', 'orange'])
    
        # Adjust Y-axis limits to leave space above the bars
        ax3.set_ylim(0, max(previous_year_revenue_billion, current_year_revenue_billion) * 1.2)
    
        # Adding value labels inside of the bars at the top in white
        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval * .95, round(yval, 2), ha='center', va='top', fontsize=18, fontweight='bold', color='white')
    
        # Adding year labels inside of the bars toward the bottom
        for bar_idx, bar in enumerate(bars):
            ax3.text(bar.get_x() + bar.get_width()/2, -0.08, ["2022", "2023"][bar_idx], ha='center', va='bottom', fontsize=18, fontweight='bold', color='white')
    
        # Adding growth line with color based on direction
        ax3.plot(["2022", "2023"], [previous_year_revenue_billion, current_year_revenue_billion], color=line_color, marker='o', linestyle='-', linewidth=2)
        ax3.text(1, current_year_revenue_billion * 1.05, f"{round(growth, 2)}%", color=line_color, ha='center', va='bottom', fontsize=16)
    
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.set_xticks([])
        ax3.set_yticks([])

        # 52-Week Range (Fourth Column)
        ax4 = axs[i, 4]
        stock_data = scrape_stock_data(ticker)
        current_price = stock_data["Current Price"]
        week_low = stock_data["52W Low"]
        week_high = stock_data["52W High"]
    
        # Calculate padding for visual clarity
        padding = (week_high - week_low) * 0.05
        ax4.set_xlim(week_low - padding, week_high + padding)
    
        # Draw a horizontal line for the 52-week range
        ax4.axhline(y=0.5, xmin=0, xmax=1, color='black', linewidth=3)
    
        # Plot the Current Price as a red dot
        ax4.scatter(current_price, 0.5, color='red', s=200)
    
        # Annotations and labels
        ax4.annotate(f'${current_price:.2f}', xy=(current_price, 0.5), fontsize=16, color='red', ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')
        ax4.annotate(f'${week_low:.2f}', xy=(week_low, 0.5), fontsize=16, color='black', ha='left', va='top', xytext=(5, -20), textcoords='offset points')
        ax4.annotate(f'${week_high:.2f}', xy=(week_high, 0.5), fontsize=16, color='black', ha='right', va='top', xytext=(-5, -20), textcoords='offset points')
    
        ax4.axis('off')


    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

if st.sidebar.checkbox("Income Statement"):
    st.subheader(f"Income Statement for {selected_stock}")
    financial_statements = get_financial_statements(selected_stock)
    if financial_statements and not financial_statements['income_statement'].empty:
        st.dataframe(financial_statements['income_statement'])
    else:
        st.write("Income Statement data not available.")

if st.sidebar.checkbox("Balance Sheet"):
    st.subheader(f"Balance Sheet for {selected_stock}")
    financial_statements = get_financial_statements(selected_stock)
    if financial_statements and not financial_statements['balance_sheet'].empty:
        st.dataframe(financial_statements['balance_sheet'])
    else:
        st.write("Balance Sheet data not available.")

if st.sidebar.checkbox("Cash Flow"):
    st.subheader(f"Cash Flow for {selected_stock}")
    financial_statements = get_financial_statements(selected_stock)
    if financial_statements and not financial_statements['cash_flow'].empty:
        st.dataframe(financial_statements['cash_flow'])
    else:
        st.write("Cash Flow data not available.")

if st.sidebar.checkbox("Calculate FCFF and FCFE"):
    st.subheader(f"FCFF & FCFE for {selected_stock}")
    fcff_fcfe_results = calculate_fcff_and_fcfe(selected_stock)
    #st.write(fcff_fcfe_results)
    st.table(fcff_fcfe_results)
