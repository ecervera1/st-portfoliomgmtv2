import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from datetime import datetime
import re
from prophet import Prophet
import numpy as np
import requests
from bs4 import BeautifulSoup



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

# Function to generate Prophet forecast plot for a given stock ticker
def generate_prophet_forecast(ticker, start_date, end_date):
    # Load historical stock data
    pdata = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Prepare data for Prophet
    phdata = pdata.reset_index()
    phdata = phdata[['Date', 'Close']]
    phdata = phdata.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(phdata)

    # Create a DataFrame for future dates
    future = model.make_future_dataframe(periods=forecast_days)  # 365 Predict for 1 year into the future
    forecast = model.predict(future)

    # Plot the historical data and forecasted prices
    fig = model.plot(forecast, xlabel='Date', ylabel='Stock Price')
    plt.title(f'Historical and Forecasted Stock Prices for {ticker}')

    return fig  # Return the Prophet forecast plot as a Matplotlib figure

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data['Adj Close']
    
def calculate_parameters(data):
    returns = data.pct_change()
    mean_return = returns.mean()
    sigma = returns.std()
    return mean_return, sigma

# Function for Monte Carlo simulation
def monte_carlo_simulation(data, num_simulations=1000000, forecast_days=252):
    mean_return, sigma = calculate_parameters(data)
    final_prices = np.zeros(num_simulations)
    initial_price = data.iloc[-1]

    for i in range(num_simulations):
        random_shocks = np.random.normal(loc=mean_return, scale=sigma, size=forecast_days)
        price_series = [initial_price * (1 + random_shock) for random_shock in random_shocks]
        final_prices[i] = price_series[-1]

    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    
    # Plot the histogram
    ax.hist(final_prices, bins=50)
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Final Price Distribution after Monte Carlo Simulation')

    # Add text under the plot
    text_x = plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) * 0.02  # Adjust x-position
    text_y = plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.3  # Adjust y-position
    ax.text(text_x, text_y, f"Simulated Mean Final Price: {np.mean(final_prices):.2f}", fontsize=14)
    text_y -= (plt.ylim()[1] - plt.ylim()[0]) * 0.1  # Adjust y-position
    ax.text(text_x, text_y, f"Simulated Median Final Price: {np.median(final_prices):.2f}", fontsize=14)
    text_y -= (plt.ylim()[1] - plt.ylim()[0]) * 0.1  # Adjust y-position
    ax.text(text_x, text_y, f"Simulated Std Deviation of Final Price: {np.std(final_prices):.2f}", fontsize=14)

    # Display the Matplotlib figure using st.pyplot()
    st.pyplot(fig)

    return final_prices





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
        "Earnings Growth": info.get("earningsGrowth"),
        "Target Low": info.get("targetLowPrice"),
        "Target Mean": info.get("targetMeanPrice"),
        "Target Median": info.get("targetMedianPrice"),
        "Recommendation Mean": info.get("recommendationMean"),
        "Recommendation Key": info.get("recommendationKey")
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
user_input = st.sidebar.text_input("Enter stock tickers separated by commas", "LLY, ABT, MRNA, JNJ, BIIB, BMY, PFE, NVO, UNH")
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


stock_data_type = {}
for ticker in tickers:
    stock_data_type[ticker] = scrape_stock_data(ticker)

# Filter out only equities
equity_tickers = [ticker for ticker, data in stock_data_type.items() if data.get('quoteType') == 'EQUITY']



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

#if st.sidebar.checkbox("Calculate FCFF and FCFE"):
    #st.subheader(f"FCFF & FCFE for {selected_stock}")
    #fcff_fcfe_results = calculate_fcff_and_fcfe(selected_stock)
    #st.write(fcff_fcfe_results)
    #st.table(fcff_fcfe_results)

#Adding news 2/5/2024

if st.sidebar.checkbox("News & Articles"):
    st.subheader('News & Articles', divider='rainbow')
    st.subheader(f":newspaper: Headlines for {selected_stock} ")
    st.markdown("")
    stock_symbol = selected_stock
    news_url = f"https://finance.yahoo.com/quote/{stock_symbol}"

    # Send a GET request to the news URL
    response = requests.get(news_url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract and print the news headlines and links
        headline_elements = soup.find_all("h3", class_="Mb(5px)")
        
        for index, headline_element in enumerate(headline_elements, start=1):
            headline_text = headline_element.get_text()
            article_link = headline_element.find('a')['href']
            full_article_link = f"https://finance.yahoo.com{article_link}"
            
            # Display the headline with a hyperlink
            st.markdown(f"{index}. - [{headline_text}]({full_article_link})")
    else:
        # Print an error message if the request fails
        st.markdown("Failed to retrieve data from Yahoo Finance.")





#Adding prophet 2/5/2024


    
# Checkbox to add Prophet forecast plot
if st.sidebar.checkbox('Add Pricing Forecast', value=False):
    
    #selected_stock_prophet = st.sidebar.selectbox("Select a Stock for Predicted Forecast", tickers)
    selected_stock_prophet = selected_stock
    st.title(f'Forecast for {selected_stock_prophet}')
    
    #sliders:
    num_runs = st.slider('Number of simulation runs: ', 5000, 1000000, 10000, 1000)
    
    forecast_days = st.slider('Days to forecast: ', 30, 504, 252, 7)
    st.write("*Please note*:")
    st.write("*The slider is used for both charts. The first is based on calendar days (365 = 1yr) and the second on trading days (252 = 1yr)*.")
    #st.write("Forecast Days: ", num_runs)
    
    if selected_stock_prophet:
        st.subheader(f'Prophet Forecast for {selected_stock_prophet}')
        start_date_prophet = st.sidebar.date_input("Start Date for Forecast", pd.to_datetime("2019-01-01"))
        end_date_prophet = st.sidebar.date_input("End Date for Forecast", default_end_date)
        
        # Call the function with the specified start_date and end_date
        st.pyplot(generate_prophet_forecast(selected_stock_prophet, start_date_prophet, end_date_prophet))

        #st.subheader('', divider='rainbow')
        st.markdown("")
        st.subheader('More Simulation Results')
        
        # Prepare data for Monte Carlo simulation
        data_mc = fetch_data(selected_stock, start_date_prophet, end_date_prophet)
        
        # Perform Monte Carlo simulation
        final_prices = monte_carlo_simulation(data_mc)
        
        # Display Monte Carlo simulation results
        st.write(f"Simulated Mean Final Price: {np.mean(final_prices):.2f}")
        st.write(f"Simulated Median Final Price: {np.median(final_prices):.2f}")
        st.write(f"Simulated Std Deviation of Final Price: {np.std(final_prices):.2f}")

        

        # Call the function with the specified data
        #final_prices = monte_carlo_simulation(data_mc, num_simulations=num_runs, forecast_days=forecast_days)
        
        # Display the histogram plot
        #st.pyplot()

if st.sidebar.checkbox('Portflio', value=False):
    # Password for access
    correct_password = "ud"
    # Create a checkbox to toggle password visibility
    show_password = st.checkbox("Show Password")
    # Create an input box for the password
    password_input = st.text_input("Enter Password", type="password")
    
    # Check if the password is correct
    if password_input == correct_password:
        def get_industry(symbol):
            try:
                stock_info = yf.Ticker(symbol).info
                industry = stock_info.get("sector", "Treasury")
                return industry
            except Exception as e:
                print(f"Error fetching industry for {symbol}: {str(e)}")
                return "Error"
    
        # Function to load the data and add industry information
        def load_data():
            # Load your data here
            df = pd.read_csv('SMIF Portfolio Positions_03042024.csv')
            #df = pd.read_csv('Portfolio Positions_02092024.xlsx', usecols=lambda col: col != 'Unnamed: 0')
        
            # Fetch the industry for each symbol and add it as a column
            df['Industry'] = df['Symbol'].apply(get_industry)
            return df
        
        # Streamlit script starts here
        st.title('Portfolio')
        
        # Load data with industry information
        df = load_data()
        selected_columns = ['Symbol', 'Description', 'Current Value', 'Percent Of Account', 'Quantity', 'Cost Basis Total', 'Industry']
        #df = df[selected_columns]
        #df = df.iloc[1::2, :][selected_columns]
    
        condition = df['Quantity'].notnull()
        df = df.loc[condition, selected_columns]
    
    
        # Apply the function to split the column into three columns
        #df[['Middle Part', 'Dollar Amount', 'Percentage']] = df['Current Value % of Account'].apply(split_current_value).apply(pd.Series)
        #new_column_names = {'Middle Part': 'Current Value', 'Dollar Amount': 'Cost', 'Percentage': 'Percentage of Portfolio'}
        #df.rename(columns=new_column_names, inplace=True)
        
        #df=df.reset_index(drop=True)
        #df=df.iloc[:, [0,2,3,4,5,6]]
        #df['Percentage of Portfolio'] = df['Percentage of Portfolio'].apply(lambda x: "{:.0%}".format(x))
    
        st.dataframe(df)
        
        # Filter UI
        #industries = df['Industry'].unique()
        #selected_industry = st.selectbox('Select Industry', ['All'] + list(industries))  # Add 'All' as an option
        
        # Filtering data based on selection
        #if selected_industry == 'All':
            #filtered_data = df  # Show all data
        #else:
            #filtered_data = df[df['Industry'] == selected_industry]
        
        # Displaying filtered data
        #st.dataframe(filtered_data)

        #df['Percent Of Account'] = pd.to_numeric(df['Percent Of Account'], errors='coerce')
        df['Percent Of Account'] = df['Percent Of Account'].str.replace('%', '').astype(float)
        industry_percentages = df['Percent Of Account'].groupby(df['Industry']).sum() / df['Percent Of Account'].sum()
        symbol_percentages = df['Percent Of Account'].groupby(df['Symbol']).sum() / df['Percent Of Account'].sum()
        
        # Create a pie chart for industries
        plt.figure(figsize=(8, 8))
        plt.pie(industry_percentages, labels=industry_percentages.index, autopct='%1.1f%%', startangle=140)
        plt.title('Industries as % of Portfolio')
        plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular
        
        # Display the pie chart for industries
        plt.show()
        
        # Create a pie chart for symbols
        plt.figure(figsize=(8, 8))
        #plt.pie(symbol_percentages, labels=df['Symbol'], autopct='%1.1f%%', startangle=140)
        plt.title('Symbols as % of Portfolio')
        plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular
        
        # Display the pie chart for symbols
        plt.show()
    
        #---------------
        st.sidebar.title('Portfolio Analysis')
        selected_chart = st.sidebar.radio('Select Chart:', ['Industries', 'Ticker'])
    
        # Display the selected chart
        if selected_chart == 'Industries':
            st.title('Industries as % of Portfolio')
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(industry_percentages, labels=industry_percentages.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular
            st.pyplot(fig)
            
        else:
            st.title('Symbols as % of Portfolio')
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(symbol_percentages, labels=df['Symbol'], autopct='%1.1f%%', startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular
            st.pyplot(fig)
    else:
        st.error("Wrong password. Please try again.")
        
# Portfolio Optimizer

def run_analysis(tickers, start_date, end_date):
    # Initialize data as an empty DataFrame
    data = pd.DataFrame()

    # Fetch historical closing prices for valid tickers
    valid_tickers = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            if not df.empty:
                data = pd.concat([data, df], axis=1)
                valid_tickers.append(ticker)
            else:
                st.warning(f"No data available for {ticker}. Skipping...")
        except Exception as e:
            st.error(f"Failed to fetch data for {ticker}: {e}")

    if data.empty:
        st.error("No valid data found for any ticker symbols.")
        return

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()

    # Covariance matrix
    cov_matrix = daily_returns.cov()

    # Define the function to be minimized (negative Sharpe ratio)
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, daily_returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = portfolio_return / portfolio_volatility
        return -sharpe_ratio

    # Define the bounds for the weights
    bounds = [(0, 1) for _ in range(len(valid_tickers))]

    # Define the constraints for the weights (sum of weights equals 1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Calculate the optimal weights
    initial_guess = [1. / len(valid_tickers) for _ in range(len(valid_tickers))]
    optimal_weights = minimize(negative_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Display ticker weights
    ticker_weights = dict(zip(valid_tickers, optimal_weights.x))
    st.write("### Suggested Ticker Weights")
    st.table(pd.DataFrame.from_dict(ticker_weights, orient='index', columns=['Weight']))

    # Portfolio Statistics
    optimal_portfolio_return = np.dot(optimal_weights.x, daily_returns.mean()) * 252
    optimal_portfolio_volatility = np.sqrt(np.dot(optimal_weights.x.T, np.dot(cov_matrix, optimal_weights.x))) * np.sqrt(252)
    sharpe_ratio = optimal_portfolio_return / optimal_portfolio_volatility

    # Display portfolio statistics
    st.write(f'**Annual Return:** {optimal_portfolio_return:.2f}')
    st.write(f'**Daily Return:** {np.dot(optimal_weights.x, daily_returns.mean()):.4f}')
    st.write(f'**Risk (Standard Deviation):** {optimal_portfolio_volatility:.2f}')
    st.write(f'**Sharpe Ratio:** {sharpe_ratio:.2f}')

    # Plotting the efficient frontier
    port_returns = []
    port_volatility = []

    num_assets = len(valid_tickers)
    num_portfolios = 5000

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(daily_returns.mean(), weights) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        port_returns.append(returns)
        port_volatility.append(volatility)

    # Plotting the efficient frontier
    plt.figure(figsize=(10, 8))
    plt.scatter(port_volatility, port_returns, c=np.array(port_returns) / np.array(port_volatility), cmap='YlGnBu')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    st.pyplot(plt)

if st.sidebar.checkbox('Portflio Optimizer', value=False):
    st.title("Portfolio Optimization")
    run_button = st.checkbox("Run Analysis")

    if run_button:
        st.header("Input Parameters")
        tickers = st.text_input("Enter tickers separated by commas", "AAPL,MSFT,TSLA")
        start_date = st.text_input("Start Date (YYYY-MM-DD)", "2014-01-01")
        end_date = st.text_input("End Date (YYYY-MM-DD)", "2024-02-12")
        execute_button = st.button("Execute Analysis")

        if execute_button:
            run_analysis(tickers.split(','), start_date, end_date)
    
        
    







    
