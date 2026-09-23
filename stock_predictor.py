import difflib
import sys

import akshare as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


FEATURES = ["Return", "MA5", "MA20", "MA_Diff", "Volume_Change", "RSI"]

STOCK_LIST = [
    # Big technology and AI
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "QCOM",
    "INTC", "IBM", "ORCL", "ADBE", "CRM", "NOW", "CSCO", "SNOW", "PLTR", "ARM",

    # Semiconductor companies
    "TSM", "ASML", "AMAT", "LRCX", "KLAC", "MRVL", "MU", "NXPI", "ADI", "MCHP",
    "MPWR", "ON", "TER", "ENTG", "COHR", "GFS", "WOLF", "SWKS", "QRVO", "LSCC",

    # Cybersecurity and cloud computing
    "CRWD", "PANW", "ZS", "OKTA", "DDOG", "NET", "FTNT", "CYBR", "MDB", "DOCU",
    "TEAM", "HUBS", "WDAY", "ANET", "VRSN", "CDNS", "SNPS", "INTU",

    # Internet and media platforms
    "UBER", "ABNB", "BKNG", "ROKU", "SPOT", "TTD", "EA", "DIS", "PINS", "SNAP",
    "MTCH", "LYV", "CHTR", "CMCSA", "WBD", "PARA", "FOXA",

    # Financial technology and payments
    "V", "MA", "PYPL", "SQ", "AXP", "COIN", "HOOD", "SOFI", "UPST", "AFRM",
    "DFS", "SYF",

    # Banks and financial services
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "USB", "PNC",
    "BK", "TROW", "RJF", "AFL", "ALL", "CB", "PGR", "AIG", "MET", "TRV",
    "MMC", "SPGI", "MCO",

    # Consumer companies
    "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "PEP", "KO",
    "PG", "EL", "ULTA", "LULU", "CMG", "YUM", "DPZ", "SYY", "DG", "DLTR",
    "ROST", "TJX", "EBAY", "ETSY", "MELI", "PDD", "BABA", "JD", "CVNA",
    "CHWY", "BBY", "KR", "KDP", "MNST", "HSY", "KHC", "CL", "KMB", "GIS",

    # Healthcare companies
    "JNJ", "PFE", "MRK", "LLY", "ABBV", "TMO", "DHR", "ISRG", "VRTX", "REGN",
    "AMGN", "GILD", "BMY", "CVS", "UNH", "CI", "HUM", "MDT", "SYK", "ZBH",
    "BSX", "ILMN", "BIIB", "MRNA", "BNTX", "IDXX", "DXCM", "EW", "RMD", "ALGN",
    "HOLX", "GEHC", "IQV", "WST", "MTD", "A", "TECH", "PODD", "COR", "CAH", "MCK",

    # Industrial companies
    "CAT", "BA", "GE", "HON", "RTX", "LMT", "NOC", "UPS", "FDX", "DE",
    "ETN", "PH", "ITW", "CSX", "UNP", "NSC", "GD", "TXT", "IR", "JCI",
    "CMI", "PCAR", "PWR", "URI", "TT", "ROK", "EMR", "XYL", "FAST", "ODFL", "EXPD",

    # Airlines
    "DAL", "UAL", "AAL", "LUV",

    # Energy companies
    "XOM", "CVX", "COP", "SLB", "OXY", "PSX", "MPC", "VLO", "EOG", "DVN",
    "FANG", "APA", "CTRA", "HAL", "BKR",

    # Materials companies
    "FCX", "NEM", "LIN", "APD", "SHW",

    # Utility companies
    "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "ED", "D",

    # Real estate investment trusts
    "AMT", "PLD", "CCI", "EQIX", "DLR", "O", "SPG", "VICI",

    # Chinese and growth companies
    "NIO", "LI", "XPEV", "BIDU", "BILI", "BEKE", "TME", "NTES", "WB", "ZTO", "YUMC",

    # Electric vehicle companies
    "RIVN", "LCID",

    # Other companies
    "MP", "HCA", "MAR", "HLT", "RCL", "CCL",
]


def clean_stock_data(data, start_date):
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()
    data.index = pd.to_datetime(data.index)

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data.index.name = "Date"

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    if not all(column in data.columns for column in required_columns):
        return pd.DataFrame()

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required_columns)
    data = data[(data["Close"] > 0) & (data["Volume"] >= 0)]
    data = data[~data.index.duplicated(keep="last")].sort_index()

    return data.loc[
        data.index >= pd.Timestamp(start_date),
        required_columns
    ]


def download_from_sina(ticker, start_date):
    try:
        data = ak.stock_us_daily(symbol=ticker, adjust="qfq")

        if data is None or data.empty:
            return pd.DataFrame()

        data = data.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        if "Date" not in data.columns:
            return pd.DataFrame()

        data = data.set_index("Date")
        return clean_stock_data(data, start_date)

    except Exception as error:
        print(f"Sina download failed for {ticker}: {error}")
        return pd.DataFrame()


def download_from_yahoo(ticker, start_date):
    try:
        data = yf.Ticker(ticker).history(
            start=start_date,
            interval="1d",
            auto_adjust=False,
            actions=False,
            timeout=20,
        )

        return clean_stock_data(data, start_date)

    except Exception as error:
        print(f"Yahoo download failed for {ticker}: {error}")
        return pd.DataFrame()


def download_stock_data(ticker, start_date="2020-01-01"):
    data = download_from_sina(ticker, start_date)

    if not data.empty:
        print(f"{ticker}: data source is Sina Finance.")
        return data, "Sina Finance"

    print(f"{ticker}: trying Yahoo Finance as a backup source...")

    data = download_from_yahoo(ticker, start_date)

    if not data.empty:
        print(f"{ticker}: data source is Yahoo Finance.")
        return data, "Yahoo Finance"

    print(f"{ticker}: no data could be downloaded.")
    return pd.DataFrame(), None


def add_features(data):
    # Add technical indicators used by the prediction models.
    data = data.copy()

    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA_Diff"] = data["MA5"] - data["MA20"]
    data["Volume_Change"] = data["Volume"].pct_change()

    price_change = data["Close"].diff()
    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.rolling(14).mean()
    average_loss = loss.rolling(14).mean()

    relative_strength = average_gain / average_loss
    data["RSI"] = 100 - (100 / (1 + relative_strength))

    return data


def prepare_data(data):
    data = add_features(data)

    data["Target_1d"] = np.where(
        data["Return"].shift(-1) > 0.003,
        1,
        0
    )

    future_return = data["Close"].shift(-5) / data["Close"] - 1

    data["Target_5d"] = np.where(
        future_return > 0.01,
        1,
        0
    )

    return data.replace([np.inf, -np.inf], np.nan).dropna()


def train_models(data):
    features = data[FEATURES]

    split_index = int(len(data) * 0.7)

    train_features = features.iloc[:split_index]
    test_features = features.iloc[split_index:]

    train_target_1d = data["Target_1d"].iloc[:split_index]
    test_target_1d = data["Target_1d"].iloc[split_index:]

    train_target_5d = data["Target_5d"].iloc[:split_index]
    test_target_5d = data["Target_5d"].iloc[split_index:]

    model_settings = {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
    }

    model_1d = RandomForestClassifier(**model_settings)
    model_5d = RandomForestClassifier(**model_settings)

    model_1d.fit(train_features, train_target_1d)
    model_5d.fit(train_features, train_target_5d)

    predictions_1d = model_1d.predict(test_features)
    predictions_5d = model_5d.predict(test_features)

    return {
        "model_1d": model_1d,
        "model_5d": model_5d,
        "test_data": data.loc[test_features.index].copy(),
        "predictions_1d": predictions_1d,
        "accuracy_1d": accuracy_score(test_target_1d, predictions_1d),
        "accuracy_5d": accuracy_score(test_target_5d, predictions_5d),
    }


def get_up_probability(model, feature_row):
    probabilities = model.predict_proba(feature_row)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        return 0.0

    return float(probabilities[classes.index(1)])


def predict_stock(data, model_1d, model_5d):
    latest_features = data[FEATURES].iloc[-1:]

    return {
        "direction_1d": int(model_1d.predict(latest_features)[0]),
        "up_probability_1d": get_up_probability(model_1d, latest_features),
        "direction_5d": int(model_5d.predict(latest_features)[0]),
        "up_probability_5d": get_up_probability(model_5d, latest_features),
    }


def calculate_backtest(test_data, predictions):
    market_returns = test_data["Return"].shift(-1)

    strategy_returns = market_returns * predictions

    market_returns = market_returns.dropna()
    strategy_returns = strategy_returns.loc[market_returns.index]

    predictions = predictions[:-1]

    market_curve = (1 + market_returns).cumprod()
    strategy_curve = (1 + strategy_returns).cumprod()

    total_trades = int(predictions.sum())
    winning_trades = int((strategy_returns > 0).sum())

    win_rate = (
        winning_trades / total_trades
        if total_trades
        else 0.0
    )

    drawdown = strategy_curve / strategy_curve.cummax() - 1

    standard_deviation = strategy_returns.std()

    sharpe_ratio = (
        strategy_returns.mean() / standard_deviation
        if standard_deviation != 0
        else 0.0
    )

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "strategy_return": strategy_curve.iloc[-1] - 1,
        "market_return": market_curve.iloc[-1] - 1,
        "max_drawdown": drawdown.min(),
        "sharpe_ratio": sharpe_ratio,
        "market_curve": market_curve,
        "strategy_curve": strategy_curve,
        "buy_signals": predictions == 1,
        "sell_signals": predictions == 0,
    }


def plot_backtest(ticker, result):
    market_curve = result["market_curve"]
    strategy_curve = result["strategy_curve"]

    plt.figure(figsize=(12, 6))

    plt.plot(
        market_curve.index,
        market_curve.values,
        label="Buy & Hold"
    )

    plt.plot(
        strategy_curve.index,
        strategy_curve.values,
        label="AI Strategy"
    )

    plt.scatter(
        strategy_curve.index[result["buy_signals"]],
        strategy_curve[result["buy_signals"]],
        marker="^",
        label="AI Buy",
    )

    plt.scatter(
        strategy_curve.index[result["sell_signals"]],
        strategy_curve[result["sell_signals"]],
        marker="v",
        label="AI Sell",
    )

    plt.title(f"{ticker} AI Trading Strategy Backtest")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.show()


def get_close_matches(ticker, stock_list, number=3):
    return difflib.get_close_matches(
        ticker,
        stock_list,
        n=number,
        cutoff=0.4
    )


def confirm_ticker(user_input, stock_list, label="stock ticker"):
    ticker = user_input.strip().upper()

    while ticker not in stock_list:
        print(f"\nThe {label} {ticker} is not in the database.")

        matches = get_close_matches(ticker, stock_list)

        if not matches:
            return None

        for number, match in enumerate(matches, start=1):
            print(f"{number}. {match}")

        choice = input(
            "Enter a suggestion number, or enter N to type the ticker again: "
        ).strip().upper()

        if choice == "N":
            ticker = input(
                f"Please re-enter the {label}: "
            ).strip().upper()

        elif choice.isdigit() and 1 <= int(choice) <= len(matches):
            ticker = matches[int(choice) - 1]

        else:
            print("Invalid choice. Please try again.")

    return ticker


def read_watchlist(user_input, stock_list):
    watchlist = []

    for item in user_input.split(","):
        ticker = item.strip().upper()

        if not ticker or ticker in watchlist:
            continue

        if ticker in stock_list:
            watchlist.append(ticker)
            continue

        print(f"\nThe watchlist ticker {ticker} is not in the database.")

        matches = get_close_matches(ticker, stock_list)

        if not matches:
            print(f"No similar ticker was found for {ticker}. It was skipped.")
            continue

        for number, match in enumerate(matches, start=1):
            print(f"{number}. {match}")

        choice = input(
            f"Choose a suggestion for {ticker}, or enter S to skip: "
        ).strip().upper()

        if choice.isdigit() and 1 <= int(choice) <= len(matches):
            selected = matches[int(choice) - 1]

            if selected not in watchlist:
                watchlist.append(selected)

        else:
            print(f"Skipped {ticker}.")

    return watchlist


def print_program_title():
    print("AI Stock Prediction & Recommendation System")
    print("\nNumber of Available Stocks in Database:", len(STOCK_LIST))
    print("Examples: AAPL, NVDA, TSLA, MSFT")


def ask_main_ticker(stock_list):
    user_input = input(
        "\nEnter the stock ticker you want to analyze: "
    )

    return confirm_ticker(user_input, stock_list)


def ask_watchlist(stock_list):
    user_input = input(
        "Enter your watchlist, separated by commas "
        "(for example AAPL,NVDA,TSLA,MSFT): "
    )

    return read_watchlist(user_input, stock_list)


def is_recent_data(data, max_age_days=7):
    if data.empty:
        return False

    today = pd.Timestamp.now(
        tz="America/New_York"
    ).tz_localize(None).normalize()

    last_date = pd.Timestamp(
        data.index[-1]
    ).tz_localize(None)

    return (today - last_date).days <= max_age_days


def print_prediction_results(ticker, prediction, models):
    print("\n Your Stock Prediction Results")
    print("Stock Ticker:", ticker)
    print("1-Day Model Accuracy:", round(models["accuracy_1d"], 4))
    print("5-Day Model Accuracy:", round(models["accuracy_5d"], 4))

    direction_1d = "UP" if prediction["direction_1d"] == 1 else "DOWN"
    direction_5d = "BULLISH" if prediction["direction_5d"] == 1 else "BEARISH"

    print(f"\nTomorrow Forecast: {direction_1d}")

    print(
        "Probability of Increase Tomorrow:",
        f"{prediction['up_probability_1d']:.2%}"
    )

    print(
        "Probability of Decrease Tomorrow:",
        f"{1 - prediction['up_probability_1d']:.2%}"
    )

    print(f"\n5-Day Forecast: {direction_5d}")

    print(
        "Probability of Increase in 5 Days:",
        f"{prediction['up_probability_5d']:.2%}"
    )

    print(
        "Probability of Decrease in 5 Days:",
        f"{1 - prediction['up_probability_5d']:.2%}"
    )


def print_backtest_results(result):
    print("\n Backtest Performance")
    print("Total AI Trades:", result["total_trades"])
    print("Win Rate:", f"{result['win_rate']:.2%}")
    print("AI Strategy Return:", f"{result['strategy_return']:.2%}")
    print("Buy & Hold Return:", f"{result['market_return']:.2%}")
    print("Maximum Drawdown:", f"{result['max_drawdown']:.2%}")
    print("Sharpe Ratio:", round(result["sharpe_ratio"], 2))


def analyze_watchlist(watchlist, main_ticker, main_history):
    recommendations = []
    analyzed_count = 0

    histories = {
        main_ticker: main_history
    }

    stock_models = {}

    today = pd.Timestamp.now(
        tz="America/New_York"
    ).tz_localize(None).normalize()

    one_year_ago = today - pd.DateOffset(years=1)

    for ticker in watchlist:

        if ticker == main_ticker:
            history = main_history

        else:
            history, _ = download_stock_data(
                ticker,
                "2020-01-01"
            )

            if not history.empty:
                histories[ticker] = history

        if history.empty:
            print(f"{ticker}: no price data; skipped.")
            continue

        if not is_recent_data(history):
            print(f"{ticker}: the newest price is too old; skipped.")
            continue

        recent_history = history.loc[
            (history.index >= one_year_ago)
            & (history.index <= today)
        ]

        prepared = prepare_data(recent_history)

        if len(prepared) < 40:
            print(f"{ticker}: not enough recent data; skipped.")
            continue

        try:
            # Train models using this stock's own historical data.
            models = train_models(prepared)

            prediction = predict_stock(
                prepared,
                models["model_1d"],
                models["model_5d"]
            )

        except (ValueError, KeyError, IndexError) as error:
            print(f"{ticker}: analysis failed; skipped: {error}")
            continue

        stock_models[ticker] = models
        analyzed_count += 1

        if (
            prediction["direction_1d"] == 1
            and prediction["direction_5d"] == 1
        ):
            score = (
                prediction["up_probability_1d"]
                + prediction["up_probability_5d"]
            ) / 2

            recommendations.append((
                ticker,
                prediction["up_probability_1d"],
                prediction["up_probability_5d"],
                score,
            ))

    return recommendations, analyzed_count, histories, stock_models


def show_recommendations(recommendations, stock_models):
    recommendations.sort(
        key=lambda item: item[3],
        reverse=True
    )

    if not recommendations:
        print("No stock recommendations are available today.")
        return

    print("\nTop Stock Recommendations:")

    for number, item in enumerate(recommendations, start=1):
        ticker, probability_1d, probability_5d, score = item

        print(
            f"{number}. {ticker} | "
            f"1-Day Up Probability: {probability_1d:.2%} | "
            f"5-Day Up Probability: {probability_5d:.2%} | "
            f"Score: {score:.2%}"
        )

    best_ticker, best_probability_1d, best_probability_5d, best_score = (
        recommendations[0]
    )

    print("\nBest Stock Recommendation:")

    print(
        f"{best_ticker} | "
        f"1-Day Up Probability: {best_probability_1d:.2%} | "
        f"5-Day Up Probability: {best_probability_5d:.2%} | "
        f"Score: {best_score:.2%}"
    )

    if len(recommendations) > 1:
        second_ticker, probability_1d, probability_5d, score = (
            recommendations[1]
        )

        print("\nRunner-Up Recommendation:")

        print(
            f"{second_ticker} | "
            f"1-Day Up Probability: {probability_1d:.2%} | "
            f"5-Day Up Probability: {probability_5d:.2%} | "
            f"Score: {score:.2%}"
        )

    choice = input(
        "\nEnter a recommended stock ticker to view its chart, "
        "or enter N to exit: "
    ).strip().upper()

    if choice == "N":
        print("Program finished.")
        return

    recommended_tickers = [
        item[0] for item in recommendations
    ]

    if choice not in recommended_tickers:
        print("That stock is not in the recommendation list.")
        return

    models = stock_models[choice]

    backtest = calculate_backtest(
        models["test_data"],
        models["predictions_1d"]
    )

    plot_backtest(
        choice,
        backtest
    )


def main():
    print_program_title()

    ticker = ask_main_ticker(STOCK_LIST)

    if ticker is None:
        print("The ticker could not be confirmed.")
        return 1

    watchlist = ask_watchlist(STOCK_LIST)

    if not watchlist:
        print("The watchlist is empty.")
        return 1

    print("\nYour watchlist:", watchlist)

    print("\nDownloading data for the main stock...")

    history, source = download_stock_data(
        ticker,
        "2020-01-01"
    )

    if history.empty:
        print("The main stock data could not be downloaded.")
        return 1

    print(f"Main stock data source: {source}")

    if not is_recent_data(history):
        print("The newest main stock price is too old for prediction.")
        return 1

    prepared = prepare_data(history)

    if len(prepared) < 100:
        print("There is not enough data to train the models.")
        return 1

    models = train_models(prepared)

    prediction = predict_stock(
        prepared,
        models["model_1d"],
        models["model_5d"]
    )

    print_prediction_results(
        ticker,
        prediction,
        models
    )

    backtest = calculate_backtest(
        models["test_data"],
        models["predictions_1d"]
    )

    print_backtest_results(backtest)

    plot_backtest(
        ticker,
        backtest
    )

    print("\n Watchlist Recommendations")

    recommendations, analyzed, histories, stock_models = analyze_watchlist(
        watchlist,
        ticker,
        history
    )

    print(
        f"Successfully analyzed {analyzed}/{len(watchlist)} watchlist stocks."
    )

    if analyzed == 0:
        print("No watchlist stock could be analyzed.")
        return 1

    show_recommendations(
        recommendations,
        stock_models
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
