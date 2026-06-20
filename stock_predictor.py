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

    print(f"\n你输入的{prompt_name} {ticker} 不存在。")

    matches = get_best_match_list(ticker, database_pool, n=3)

    if not matches:
        print("系统没有找到接近的股票代码。")
        return None

    print("你是不是想输入以下代码之一？")
    for i, match in enumerate(matches, start=1):
        print(f"{i}. {match}")

    while True:
        choice = input("请输入序号确认（1/2/3），或输入 N 重新手动输入: ").strip().upper()

        if choice == "N":
            new_ticker = input(f"请重新输入正确的{prompt_name}: ").strip().upper()

            if new_ticker in database_pool:
                print(f"你已确认使用: {new_ticker}")
                return new_ticker

            print(f"{new_ticker} 仍然不在数据库中。")
            new_matches = get_best_match_list(new_ticker, database_pool, n=3)

            if new_matches:
                print("最接近的代码有：")
                for i, match in enumerate(new_matches, start=1):
                    print(f"{i}. {match}")
            else:
                print("系统还是没有找到接近的代码。")

        elif choice in ["1", "2", "3"]:
            idx = int(choice) - 1
            if idx < len(matches):
                confirmed = matches[idx]
                print(f"你已确认使用: {confirmed}")
                return confirmed
            else:
                print("这个序号超出范围，请重新输入。")
        else:
            print("输入无效，请重新输入。")


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

        print(f"\n你输入的关注股票代码 {ticker} 不存在。")

        matches = get_best_match_list(ticker, database_pool, n=3)

        if not matches:
            print(f"{ticker} 没有找到接近的代码，已跳过。")
            continue

        print("你是不是想输入以下代码之一？")
        for i, match in enumerate(matches, start=1):
            print(f"{i}. {match}")

        while True:
            choice = input(
                f"请为 {ticker} 输入序号确认（1/2/3），输入 N 重新手动输入，或输入 S 跳过: "
            ).strip().upper()

            if choice == "S":
                print(f"已跳过 {ticker}")
                break

            elif choice == "N":
                new_ticker = input("请重新输入这个关注股票代码: ").strip().upper()

                if new_ticker in database_pool:
                    if new_ticker not in cleaned:
                        cleaned.append(new_ticker)
                    print(f"已确认加入: {new_ticker}")
                    break
                else:
                    print(f"{new_ticker} 不在数据库中。")
                    new_matches = get_best_match_list(new_ticker, database_pool, n=3)
                    if new_matches:
                        print("最接近的代码有：")
                        for i, match in enumerate(new_matches, start=1):
                            print(f"{i}. {match}")
                    else:
                        print("系统没有找到接近的代码。")

            elif choice in ["1", "2", "3"]:
                idx = int(choice) - 1
                if idx < len(matches):
                    confirmed = matches[idx]
                    if confirmed not in cleaned:
                        cleaned.append(confirmed)
                    print(f"已确认加入: {confirmed}")
                    break
                else:
                    print("这个序号超出范围，请重新输入。")
            else:
                print("输入无效，请重新输入。")

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

    print("\n数据库内可用股票数量:", len(database_pool))
    print("示例: AAPL, NVDA, TSLA, MSFT")

    stock_code_input = input("\n请输入你要重点分析的股票代码: ")
    stock_code = confirm_single_ticker(
        stock_code_input,
        database_pool,
        prompt_name="重点分析股票代码"
    )

    if stock_code is None:
        print("无法确认重点分析股票代码，程序结束。")
        return

    watchlist_input = input(
        "请输入你关注的股票列表（用逗号分隔，例如 AAPL,NVDA,TSLA,MSFT）: "
    ).strip()

    watchlist = parse_watchlist_with_confirmation(watchlist_input, database_pool)

    if len(watchlist) == 0:
        print("你确认后的关注股票列表为空。")
        return

    print("\n你的关注股票列表:", watchlist)

    # =========================
    # 下载重点分析股票数据
    # =========================
    print("\n正在下载重点分析股票数据...")
    data = yf.download(stock_code, start="2020-01-01", auto_adjust=False, progress=False)

    if data.empty:
        print("数据下载失败，请检查股票代码。")
        return

    data = prepare_data(data)

    if len(data) < 100:
        print("数据不足，无法完成训练。")
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

    print("\n========== 你的股票预测结果 ==========")
    print("股票代码:", stock_code)
    print("明天模型准确率:", round(accuracy_1d, 4))
    print("未来5天模型准确率:", round(accuracy_5d, 4))

    if prediction_1d == 1:
        print("\n明天预测: 可能上涨 📈")
    else:
        print("\n明天预测: 可能下跌 📉")
    print("明天上涨概率:", f"{up_prob_1d:.2%}")
    print("明天下跌概率:", f"{down_prob_1d:.2%}")

    if prediction_5d == 1:
        print("\n未来5天趋势: 偏上涨 📈")
    else:
        print("\n未来5天趋势: 偏下跌 📉")
    print("未来5天上涨概率:", f"{up_prob_5d:.2%}")
    print("未来5天下跌概率:", f"{down_prob_5d:.2%}")

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

    print("\n========== 回测结果（基于明天预测） ==========")
    print("AI交易次数:", total_trades)
    print("盈利交易比例:", f"{win_rate:.2%}")
    print("AI策略总收益:", f"{total_return_strategy:.2%}")
    print("Buy & Hold总收益:", f"{total_return_market:.2%}")
    print("最大回撤:", f"{max_drawdown:.2%}")
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
    print("\n========== 关注列表中的推荐结果 ==========")

    recommendations = []

    if len(watchlist) == 0:
        print("你的关注列表为空，无法推荐。")
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
        print("关注列表数据下载失败。")
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
            print("关注列表数据格式异常。")
            return

        level0_values = set(all_data.columns.get_level_values(0))
        level1_values = set(all_data.columns.get_level_values(1))

        ticker_level = None
        if watchlist[0] in level0_values:
            ticker_level = 0
        elif watchlist[0] in level1_values:
            ticker_level = 1
        else:
            print("关注列表数据格式异常。")
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
        print("今天没有推荐的股票。")
    else:
        print("\n推荐股票排名：")
        for i, (ticker, prob_1d, prob_5d, score) in enumerate(recommendations, start=1):
            print(
                f"{i}. {ticker}  |  明天上涨概率: {prob_1d:.2%}  |  未来5天上涨概率: {prob_5d:.2%}  |  综合推荐分数: {score:.2%}"
            )

        best_ticker, best_prob_1d, best_prob_5d, best_score = recommendations[0]
        #新增图像
        # =========================
        # 推荐股票单独回测图
        # =========================
        print(f"\n正在生成推荐股票 {best_ticker} 的回测图...")

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

        print("\n⭐ 今日最强推荐股票:")
        print(
            f"{best_ticker}  |  明天上涨概率: {best_prob_1d:.2%}  |  未来5天上涨概率: {best_prob_5d:.2%}  |  综合推荐分数: {best_score:.2%}"
        )

        if len(recommendations) > 1:
            second_ticker, second_prob_1d, second_prob_5d, second_score = recommendations[1]
            print("\n⭐⭐ 次强推荐股票:")
            print(
                f"{second_ticker}  |  明天上涨概率: {second_prob_1d:.2%}  |  未来5天上涨概率: {second_prob_5d:.2%}  |  综合推荐分数: {second_score:.2%}"
            )


if __name__ == "__main__":
    main()
