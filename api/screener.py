# -*- coding: utf-8 -*-
# 最終版：增加了對請求數量的限制，以防止在免費平台上超時

from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import time
import traceback
import sys

# 初始化 Flask 應用
app = Flask(__name__)

# --- 核心計算邏輯 (增加健壯性) ---

def get_stock_data(ticker_symbol):
    """為單一股票獲取計算所需的所有原始數據。"""
    try:
        stock = yf.Ticker(ticker_symbol)
        if stock.history(period="5d").empty:
            print(f"警告: {ticker_symbol} 找不到歷史數據，可能為無效代碼。", file=sys.stderr)
            return None
        
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        if financials.empty or balance_sheet.empty or cashflow.empty:
            print(f"警告: {ticker_symbol} 的財報數據不完整，將被跳過。", file=sys.stderr)
            return None
            
        return {"ticker": stock, "info": info, "financials": financials, "balance_sheet": balance_sheet, "cashflow": cashflow}
    except Exception as e:
        print(f"錯誤: 在獲取 {ticker_symbol} 的數據時失敗: {e}", file=sys.stderr)
        return None

def calculate_metrics(data):
    """從已獲取的數據中計算五個核心財務指標。"""
    ticker_symbol = data['ticker'].ticker
    try:
        info = data['info']
        financials = data['financials']
        balance_sheet = data['balance_sheet']
        cashflow = data['cashflow']
        # 因子一：ROIC
        try:
            op_income = financials.loc['Operating Income'].iloc[0]
            tax_provision = financials.loc['Tax Provision'].iloc[0]
            income_before_tax = financials.loc['Pretax Income'].iloc[0]
            tax_rate = tax_provision / income_before_tax if income_before_tax > 0 else 0
            nopat = op_income * (1 - tax_rate)
            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
            total_equity = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0]
            cash_and_equivalents = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
            invested_capital = total_debt + total_equity - cash_and_equivalents
            roic = nopat / invested_capital if invested_capital != 0 else np.nan
        except (KeyError, IndexError, ZeroDivisionError):
            roic = np.nan
        # 因子二：研發/銷售比例
        try:
            rd_expense = financials.loc['Research And Development'].iloc[0]
            revenue = financials.loc['Total Revenue'].iloc[0]
            rd_to_sales = rd_expense / revenue if revenue > 0 else np.nan
        except (KeyError, IndexError, ZeroDivisionError):
            rd_to_sales = np.nan
        # 因子三：淨債務/EBITDA
        try:
            net_debt = info.get('netDebt')
            if net_debt is None:
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                cash_and_equivalents = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
                net_debt = total_debt - cash_and_equivalents
            ebitda = info.get('ebitda')
            if ebitda is None:
                op_income = financials.loc['Operating Income'].iloc[0]
                da = cashflow.loc['Depreciation And Amortization'].iloc[0]
                ebitda = op_income + da
            net_debt_to_ebitda = net_debt / ebitda if ebitda != 0 else np.nan
        except (KeyError, IndexError, TypeError, ZeroDivisionError):
            net_debt_to_ebitda = np.nan
        # 因子四：EV/FCF
        try:
            ev = info.get('enterpriseValue')
            fcf = cashflow.loc['Free Cash Flow'].iloc[0]
            ev_to_fcf = ev / fcf if fcf != 0 else np.nan
        except (KeyError, IndexError, TypeError, ZeroDivisionError):
            ev_to_fcf = np.nan
        # 因子五：營收CAGR(3Y)
        try:
            if 'Total Revenue' in financials.index and len(financials.loc['Total Revenue']) >= 4:
                revenue_t0 = financials.loc['Total Revenue'].iloc[0]
                revenue_t3 = financials.loc['Total Revenue'].iloc[3]
                cagr = ((revenue_t0 / revenue_t3)**(1/3) - 1) if revenue_t3 > 0 else np.nan
            else:
                cagr = np.nan
        except (KeyError, IndexError, ZeroDivisionError):
            cagr = np.nan
        return {'代碼': ticker_symbol, '公司名稱': info.get('shortName', ticker_symbol), 'ROIC': roic, '研發/銷售': rd_to_sales, '淨債務/EBITDA': net_debt_to_ebitda, 'EV/FCF': ev_to_fcf, '營收CAGR(3Y)': cagr}
    except Exception as e:
        print(f"計算 {ticker_symbol} 指標時發生錯誤: {e}", file=sys.stderr)
        return None

def rank_stocks(df, weights):
    """對 DataFrame 中的股票根據因子和權重進行排名。"""
    ranked_df = df.copy()
    for factor, weight_info in weights.items():
        ranked_df[f'{factor}_排名'] = ranked_df[factor].rank(ascending=(not weight_info['higher_is_better']), na_option='bottom')
    ranked_df['綜合分'] = 0
    for factor, weight_info in weights.items():
        ranked_df['綜合分'] += ranked_df[f'{factor}_排名'] * weight_info['weight']
    final_df = ranked_df.sort_values(by='綜合分').reset_index(drop=True)
    final_df.index = final_df.index + 1
    return final_df

# --- API 端點 (Endpoint) ---
@app.route('/', methods=['POST'])
def handler():
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "無效的請求: 未提供 JSON 數據"}), 400

        tickers_to_process = request_data.get('tickers')
        factor_weights = request_data.get('weights')

        if not isinstance(tickers_to_process, list) or not tickers_to_process:
            return jsonify({"error": "無效的請求: 'tickers' 必須是一個非空的列表"}), 400
        if not isinstance(factor_weights, dict) or not factor_weights:
            return jsonify({"error": "無效的請求: 'weights' 必須是一個非空的字典"}), 400

        # --- 關鍵修正：加入股票數量限制 ---
        MAX_TICKERS = 45 # 設定一個安全的上限，以確保在 Vercel 的 60 秒超時限制內完成
        if len(tickers_to_process) > MAX_TICKERS:
            error_message = f"請求的股票數量 ({len(tickers_to_process)}) 過多。為避免在免費平台上超時，請將單次篩選的股票數量限制在 {MAX_TICKERS} 支以內。"
            return jsonify({"error": error_message}), 400

        all_metrics = []
        print(f"準備處理 {len(tickers_to_process)} 支股票...", file=sys.stderr)
        sys.stderr.flush()

        for ticker in tickers_to_process:
            print(f"正在獲取 {ticker} 的數據...", file=sys.stderr)
            sys.stderr.flush()
            
            stock_data = get_stock_data(ticker)
            if stock_data:
                metrics = calculate_metrics(stock_data)
                if metrics:
                    all_metrics.append(metrics)
            else:
                print(f"警告: 無法獲取或處理 {ticker} 的數據，已跳過。", file=sys.stderr)
                sys.stderr.flush()
            time.sleep(1) 

        if not all_metrics:
            return jsonify({"error": "未能從 yfinance 獲取任何有效的股票數據。請檢查股票代碼或稍後再試。"}), 500
            
        print("數據獲取完成，開始排名...", file=sys.stderr)
        sys.stderr.flush()
        
        metrics_df = pd.DataFrame(all_metrics).replace([np.inf, -np.inf], np.nan)
        final_ranked_df = rank_stocks(metrics_df, factor_weights)

        final_ranked_df = final_ranked_df.where(pd.notnull(final_ranked_df), None)
        result_json = final_ranked_df.to_dict(orient='records')
        
        print("篩選完成，成功返回結果。", file=sys.stderr)
        sys.stderr.flush()
        return jsonify(result_json)

    except Exception as e:
        print(f"伺服器內部發生嚴重錯誤: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return jsonify({"error": "伺服器在處理請求時發生未知內部錯誤。", "message": str(e)}), 500
