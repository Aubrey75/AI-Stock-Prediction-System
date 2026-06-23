import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import difflib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def add_features(data):
    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA_Diff"] = data["MA5"] - data["MA20"]
    data["Volume_Change"] = data["Volume"].pct_change()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    return data


def prepare_data(data):
    data = add_features(data)

    # 明天涨跌目标：明天涨幅 > 0.3%
    data["Target_1d"] = np.where(data["Return"].shift(-1) > 0.003, 1, 0)

    # 未来5天走势目标：未来5天总涨幅 > 1%
    future_return_5d = data["Close"].shift(-5) / data["Close"] - 1
    data["Target_5d"] = np.where(future_return_5d > 0.01, 1, 0)

    data = data.dropna()
    return data


def get_best_match_list(ticker, database_pool, n=3):
    return difflib.get_close_matches(ticker, database_pool, n=n, cutoff=0.4)


def confirm_single_ticker(user_input, database_pool, prompt_name="股票代码"):
    ticker = user_input.strip().upper()

    if ticker in database_pool:
        return ticker

    print(f"\nThe {prompt_name} you entered, {ticker}, does not exist.")

    matches = get_best_match_list(ticker, database_pool, n=3)

    if not matches:
        print("The system could not find any similar stock tickers.")
        return None

    print("Did you mean one of the following tickers?")
    for i, match in enumerate(matches, start=1):
        print(f"{i}. {match}")

    while True:
        choice = input("Enter a number to confirm (1/2/3), or enter N to manually type again: ").strip().upper()

        if choice == "N":
            new_ticker = input(f"Please re-enter the correct {prompt_name}: ").strip().upper()

            if new_ticker in database_pool:
                print(f"Confirmed ticker: {new_ticker}")
                return new_ticker

            print(f"{new_ticker} is still not in the database.")
            new_matches = get_best_match_list(new_ticker, database_pool, n=3)

            if new_matches:
                print("Closest matching tickers:")
                for i, match in enumerate(new_matches, start=1):
                    print(f"{i}. {match}")
            else:
                print("The system still could not find any similar tickers.")

        elif choice in ["1", "2", "3"]:
            idx = int(choice) - 1
            if idx < len(matches):
                confirmed = matches[idx]
                print(f"Confirmed ticker: {confirmed}")
                return confirmed
            else:
                print("This number is out of range. Please try again.")
        else:
            print("Invalid input. Please try again.")


def parse_watchlist_with_confirmation(user_input, database_pool):
    raw_list = user_input.split(",")
    cleaned = []

    for item in raw_list:
        ticker = item.strip().upper()

        if not ticker:
            continue

        if ticker in cleaned:
            continue

        if ticker in database_pool:
            cleaned.append(ticker)
            continue

        print(f"\nThe watchlist ticker you entered, {ticker}, does not exist.")

        matches = get_best_match_list(ticker, database_pool, n=3)

        if not matches:
            print(f"No similar tickers were found for {ticker}. Skipping...")
            continue

        print("Did you mean one of the following tickers?")
        for i, match in enumerate(matches, start=1):
            print(f"{i}. {match}")

        while True:
            choice = input(
                f"For {ticker}, enter a number to confirm (1/2/3), enter N to manually re-enter, or enter S to skip: "
            ).strip().upper()

            if choice == "S":
                print(f"Skipped {ticker}")
                break

            elif choice == "N":
                new_ticker = input("Please re-enter the watchlist ticker: ").strip().upper()

                if new_ticker in database_pool:
                    if new_ticker not in cleaned:
                        cleaned.append(new_ticker)
                    print(f"Added to watchlist: {new_ticker}")
                    break
                else:
                    print(f"{new_ticker} is not in the database.")

                    new_matches = get_best_match_list(new_ticker, database_pool, n=3)

                    if new_matches:
                        print("Closest matching tickers:")
                        for i, match in enumerate(new_matches, start=1):
                            print(f"{i}. {match}")
                    else:
                        print("The system could not find any similar tickers.")

            elif choice in ["1", "2", "3"]:
                idx = int(choice) - 1

                if idx < len(matches):
                    confirmed = matches[idx]

                    if confirmed not in cleaned:
                        cleaned.append(confirmed)

                    print(f"Added to watchlist: {confirmed}")
                    break
                else:
                    print("This number is out of range. Please try again.")

            else:
                print("Invalid input. Please try again.")

    return cleaned


def main():
    print("AI Stock Prediction & Recommendation System")

    # 扩大后的股票数据库
    database_pool = [

        # Big Tech / AI
        "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AMD","AVGO","QCOM",
        "INTC","IBM","ORCL","ADBE","CRM","NOW","CSCO","SNOW","PLTR","ARM",

        # Semiconductor
        "TSM","ASML","AMAT","LRCX","KLAC","MRVL","MU","NXPI","ADI","MCHP",
        "MPWR","ON","TER","ENTG","COHR","GFS","WOLF","SWKS","QRVO","LSCC",

        # Cybersecurity / Cloud
        "CRWD","PANW","ZS","OKTA","DDOG","NET","FTNT","CYBR","MDB","DOCU",
        "TEAM","HUBS","WDAY","ANET","VRSN","CDNS","SNPS","INTU",

        # Internet / Platforms
        "UBER","ABNB","BKNG","ROKU","SPOT","TTD","EA","DIS","PINS","SNAP",
        "MTCH","LYV","CHTR","CMCSA","WBD","PARA","FOXA",

        # Fintech / Payments
        "V","MA","PYPL","SQ","AXP","COIN","HOOD","SOFI","UPST","AFRM",
        "DFS","SYF",

        # Financials
        "JPM","BAC","WFC","C","GS","MS","BLK","SCHW","USB","PNC",
        "BK","TROW","RJF","AFL","ALL","CB","PGR","AIG","MET","TRV",
        "MMC","SPGI","MCO",

        # Consumer
        "WMT","COST","TGT","HD","LOW","NKE","SBUX","MCD","PEP","KO",
        "PG","EL","ULTA","LULU","CMG","YUM","DPZ","SYY","DG","DLTR",
        "ROST","TJX","EBAY","ETSY","MELI","PDD","BABA","JD","CVNA",
        "CHWY","BBY","KR","KDP","MNST","HSY","KHC","CL","KMB","GIS",

        # Healthcare
        "JNJ","PFE","MRK","LLY","ABBV","TMO","DHR","ISRG","VRTX","REGN",
        "AMGN","GILD","BMY","CVS","UNH","CI","HUM","MDT","SYK","ZBH",
        "BSX","ILMN","BIIB","MRNA","BNTX","IDXX","DXCM","EW","RMD","ALGN",
        "HOLX","GEHC","IQV","WST","MTD","A","TECH","PODD","COR","CAH","MCK",

        # Industrials
        "CAT","BA","GE","HON","RTX","LMT","NOC","UPS","FDX","DE",
        "ETN","PH","ITW","CSX","UNP","NSC","GD","TXT","IR","JCI",
        "CMI","PCAR","PWR","URI","TT","ROK","EMR","XYL","FAST","ODFL","EXPD",

        # Airlines
        "DAL","UAL","AAL","LUV",

        # Energy
        "XOM","CVX","COP","SLB","OXY","PSX","MPC","VLO","EOG","DVN",
        "FANG","APA","CTRA","HAL","BKR",

        # Materials
        "FCX","NEM","LIN","APD","SHW",

        # Utilities
        "NEE","DUK","SO","AEP","EXC","XEL","ED","D",

        # REITs
        "AMT","PLD","CCI","EQIX","DLR","O","SPG","VICI",

        # China / Growth
        "NIO","LI","XPEV","BIDU","BILI","BEKE","TME","NTES","WB","ZTO","YUMC",

        # EV
        "RIVN","LCID",

        # Misc
        "MP","HCA","MAR","HLT","RCL","CCL",
    ]

    print("\nNumber of Available Stocks in Database:", len(database_pool))
    print("Examples: AAPL, NVDA, TSLA, MSFT")

    stock_code_input = input("\nEnter the stock ticker you want to analyze: ")

    stock_code = confirm_single_ticker(
        stock_code_input,
        database_pool,
        prompt_name="stock ticker"
    )

    if stock_code is None:
        print("Unable to confirm the stock ticker. Program terminated.")
        return

    watchlist_input = input(
        "Enter your watchlist (comma separated):(Separated by commas, for example AAPL,NVDA,TSLA,MSFT）: "
    ).strip()

    watchlist = parse_watchlist_with_confirmation(watchlist_input, database_pool)

    if len(watchlist) == 0:
        print("The list of stocks you follow after confirmation is empty.")
        return

    print("\nYour list of stocks to follow：", watchlist)

    # =========================
    # 下载重点分析股票数据
    # =========================
    print("\nKey analysis stock data is being downloaded...")
    data = yf.download(stock_code, start="2020-01-01", auto_adjust=False, progress=False)

    if data.empty:
        print("Data download failed. Please check the stock code.")
        return

    data = prepare_data(data)

    if len(data) < 100:
        print("The data is insufficient to complete the training")
        return

    features = ["Return", "MA5", "MA20", "MA_Diff", "Volume_Change", "RSI"]

    X = data[features]
    y1 = data["Target_1d"]
    y5 = data["Target_5d"]

    # =========================
    # 明天预测模型
    # =========================
    X_train_1d, X_test_1d, y_train_1d, y_test_1d = train_test_split(
        X, y1, test_size=0.3, shuffle=False
    )

    model_1d = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model_1d.fit(X_train_1d, y_train_1d)

    pred_1d_test = model_1d.predict(X_test_1d)
    accuracy_1d = accuracy_score(y_test_1d, pred_1d_test)

    # =========================
    # 未来5天预测模型
    # =========================
    X_train_5d, X_test_5d, y_train_5d, y_test_5d = train_test_split(
        X, y5, test_size=0.3, shuffle=False
    )

    model_5d = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model_5d.fit(X_train_5d, y_train_5d)

    pred_5d_test = model_5d.predict(X_test_5d)
    accuracy_5d = accuracy_score(y_test_5d, pred_5d_test)

    # =========================
    # 重点分析股票预测
    # =========================
    latest_data = X.iloc[-1:]

    prediction_1d = model_1d.predict(latest_data.values)[0]
    probability_1d = model_1d.predict_proba(latest_data.values)[0]
    down_prob_1d = probability_1d[0]
    up_prob_1d = probability_1d[1]

    prediction_5d = model_5d.predict(latest_data.values)[0]
    probability_5d = model_5d.predict_proba(latest_data.values)[0]
    down_prob_5d = probability_5d[0]
    up_prob_5d = probability_5d[1]

    print("\n========== Your stock prediction results ==========")
    print("Stock Ticker:", stock_code)
    print("1-Day Model Accuracy:", round(accuracy_1d, 4))
    print("5-Day Model Accuracy:", round(accuracy_5d, 4))

    if prediction_1d == 1:
        print("\nTomorrow Forecast: UP ")
    else:
        print("\nTomorrow Forecast: DOWN ")

    print("Probability of Increase Tomorrow:", f"{up_prob_1d:.2%}")
    print("Probability of Decrease Tomorrow:", f"{down_prob_1d:.2%}")

    if prediction_5d == 1:
        print("\n5-Day Forecast: BULLISH ")
    else:
        print("\n5-Day Forecast: BEARISH ")

    print("Probability of Increase in 5 Days:", f"{up_prob_5d:.2%}")
    print("Probability of Decrease in 5 Days:", f"{down_prob_5d:.2%}")

    # =========================
    # 回测（基于明天预测）
    # =========================
    test_returns = data["Return"].iloc[-len(pred_1d_test):]
    strategy_returns = test_returns * pred_1d_test

    cumulative_market = (1 + test_returns).cumprod()
    cumulative_strategy = (1 + strategy_returns).cumprod()

    total_trades = int(pred_1d_test.sum())
    win_trades = int((strategy_returns > 0).sum())
    win_rate = (win_trades / total_trades) if total_trades > 0 else 0.0

    total_return_strategy = cumulative_strategy.iloc[-1] - 1
    total_return_market = cumulative_market.iloc[-1] - 1

    rolling_max = cumulative_strategy.cummax()
    drawdown = cumulative_strategy / rolling_max - 1
    max_drawdown = drawdown.min()

    if strategy_returns.std() != 0:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std()
    else:
        sharpe_ratio = 0.0

    print("\n========== Backtest Performance ==========")
    print("Total AI Trades:", total_trades)
    print("Win Rate:", f"{win_rate:.2%}")
    print("AI Strategy Return:", f"{total_return_strategy:.2%}")
    print("Buy & Hold Return:", f"{total_return_market:.2%}")
    print("Maximum Drawdown:", f"{max_drawdown:.2%}")
    print("Sharpe Ratio:", round(sharpe_ratio, 2))

    # =========================
    # 图表
    # =========================
    buy_signals = pred_1d_test == 1
    sell_signals = pred_1d_test == 0

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_market.index, cumulative_market.values, label="Buy & Hold")
    plt.plot(cumulative_strategy.index, cumulative_strategy.values, label="AI Strategy")

    plt.scatter(
        cumulative_strategy.index[buy_signals],
        cumulative_strategy[buy_signals],
        marker="^",
        label="AI Buy"
    )

    plt.scatter(
        cumulative_strategy.index[sell_signals],
        cumulative_strategy[sell_signals],
        marker="v",
        label="AI Sell"
    )

    plt.title(stock_code + " AI Trading Strategy Backtest")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.show()

    # =========================
    # 推荐股票（只从用户关注列表中推荐）
    # =========================
    print("\n========== Follow the recommended results in the list ==========")

    recommendations = []

    if len(watchlist) == 0:
        print("Your follow list is empty and cannot be recommended.")
        return

    try:
        all_data = yf.download(
            tickers=watchlist,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception:
        print("The download of the follow list data failed。")
        return

    if len(watchlist) == 1:
        single_ticker = watchlist[0]
        d = all_data.copy()

        if not d.empty and len(d) >= 40:
            d = prepare_data(d)

            if len(d) > 0:
                latest = d[features].iloc[-1:]

                pred_direction_1d = model_1d.predict(latest.values)[0]
                up_probability_1d = model_1d.predict_proba(latest.values)[0][1]

                pred_direction_5d = model_5d.predict(latest.values)[0]
                up_probability_5d = model_5d.predict_proba(latest.values)[0][1]

                if pred_direction_1d == 1 and pred_direction_5d == 1:
                    score = (up_probability_1d + up_probability_5d) / 2
                    recommendations.append(
                        (single_ticker, up_probability_1d, up_probability_5d, score)
                    )

    else:
        if not isinstance(all_data.columns, pd.MultiIndex):
            print("Pay attention to the abnormal format of the list data.")
            return

        level0_values = set(all_data.columns.get_level_values(0))
        level1_values = set(all_data.columns.get_level_values(1))

        ticker_level = None
        if watchlist[0] in level0_values:
            ticker_level = 0
        elif watchlist[0] in level1_values:
            ticker_level = 1
        else:
            print("Pay attention to the abnormal format of the list data.")
            return

        for ticker in watchlist:
            try:
                d = all_data.xs(ticker, axis=1, level=ticker_level).copy()

                if d.empty or len(d) < 40:
                    continue

                d = prepare_data(d)

                if len(d) == 0:
                    continue

                latest = d[features].iloc[-1:]

                pred_direction_1d = model_1d.predict(latest)[0]
                up_probability_1d = model_1d.predict_proba(latest)[0][1]

                pred_direction_5d = model_5d.predict(latest)[0]
                up_probability_5d = model_5d.predict_proba(latest)[0][1]

                if pred_direction_1d == 1 and pred_direction_5d == 1:
                    score = (up_probability_1d + up_probability_5d) / 2
                    recommendations.append(
                        (ticker, up_probability_1d, up_probability_5d, score)
                    )

            except Exception:
                continue

    recommendations.sort(key=lambda x: x[3], reverse=True)
    if len(recommendations) == 0:
        print("No stock recommendations available today.")
    else:
        print("\nTop Stock Recommendations:")
        for i, (ticker, prob_1d, prob_5d, score) in enumerate(recommendations, start=1):
            print(
                f"{i}. {ticker}  |  1-Day Up Probability: {prob_1d:.2%}  |  5-Day Up Probability: {prob_5d:.2%}  |  Recommendation Score: {score:.2%}"
            )

        best_ticker, best_prob_1d, best_prob_5d, best_score = recommendations[0]

        # =========================
        # Backtest Chart for Top Recommendation
        # =========================
        print(f"\nGenerating backtest chart for {best_ticker}...")

        rec_data = yf.download(
            best_ticker,
            period="5y",
            auto_adjust=False,
            progress=False
        )

        if not rec_data.empty:
            rec_data = prepare_data(rec_data)

            X_rec = rec_data[features]
            y_rec = rec_data["Target_1d"]

            pred_rec = model_1d.predict(X_rec)

            rec_returns = rec_data["Return"].iloc[-len(pred_rec):]
            rec_strategy = rec_returns * pred_rec

            rec_market_curve = (1 + rec_returns).cumprod()
            rec_ai_curve = (1 + rec_strategy).cumprod()

            plt.figure(figsize=(12, 6))
            plt.plot(rec_market_curve.index, rec_market_curve.values, label="Buy & Hold")
            plt.plot(rec_ai_curve.index, rec_ai_curve.values, label="AI Strategy")

            plt.title(best_ticker + " AI Trading Strategy Backtest")
            plt.xlabel("Time")
            plt.ylabel("Cumulative Return")
            plt.grid(True)
            plt.legend()
            plt.show()

        print("\n🏆 Best Stock Recommendation:")
        print(
            f"{best_ticker}  |  1-Day Up Probability: {best_prob_1d:.2%}  |  5-Day Up Probability: {best_prob_5d:.2%}  |  Recommendation Score: {best_score:.2%}"
        )

        if len(recommendations) > 1:
            second_ticker, second_prob_1d, second_prob_5d, second_score = recommendations[1]

            print("\n🥈 Runner-Up Recommendation:")
            print(
                f"{second_ticker}  |  1-Day Up Probability: {second_prob_1d:.2%}  |  5-Day Up Probability: {second_prob_5d:.2%}  |  Recommendation Score: {second_score:.2%}"
            )



if __name__ == "__main__":
    main()
