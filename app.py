import pandas as pd
import numpy as np
import holidays
from datetime import datetime

# =====================================================================
# 1. KRX 거래일/휴장일 캘린더 (추정 캘린더 표시)
# =====================================================================
def get_krx_trading_days(year, custom_holidays=None):
    """
    KRX 개장일 캘린더를 생성합니다. (공식 API 미연동 시 추정 캘린더 표기)
    """
    kr_holidays = holidays.KR(years=year)
    known_special_holidays = {
        f"{year}-05-01": "근로자의 날",
        f"{year}-12-31": "연말 휴장일"
    }
    if custom_holidays:
        known_special_holidays.update(custom_holidays)
    for h_date, h_name in known_special_holidays.items():
        kr_holidays[h_date] = h_name

    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='B')
    trading_days = pd.DatetimeIndex([d for d in dates if d not in kr_holidays])
    
    # 캘린더 출처 한계 명시
    print(f"[Calendar Notice] {year}년 거래일 캘린더는 KRX 공식 API가 아닌 추정 캘린더입니다.")
    return trading_days


def get_next_trading_day(base_date, trading_days_dict):
    """
    기준일 다음 거래일을 찾습니다. 실패 시 (None, 사유)를 반환합니다.
    """
    base_date = pd.Timestamp(base_date).tz_localize(None).floor('D')
    current_year = base_date.year
    
    search_days = pd.DatetimeIndex([])
    for y in [current_year, current_year + 1]:
        if y in trading_days_dict:
            search_days = search_days.append(trading_days_dict[y])
            
    if search_days.empty:
        return None, f"연도 {current_year} 및 {current_year+1}의 캘린더 데이터 없음"
        
    future_days = search_days[search_days > base_date]
    if len(future_days) > 0:
        return future_days[0], "성공"
    return None, f"기준일({base_date.strftime('%Y-%m-%d')}) 이후의 거래일을 찾을 수 없음 (범위 초과)"


# =====================================================================
# 2. 데이터 정규화 및 전처리 (컬럼명 표준화 포함)
# =====================================================================
def normalize_columns(df):
    """
    기존 소문자/대문자 컬럼명을 PascalCase 표준(Open, High, Low, Close, Volume, Amount)으로 통일합니다.
    """
    df = df.copy()
    mapping = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'volume': 'Volume', 'amount': 'Amount'
    }
    rename_dict = {col: mapping[col.lower()] for col in df.columns if col.lower() in mapping}
    return df.rename(columns=rename_dict)


def validate_and_process_amount(df):
    """
    거래대금 상태를 5가지로 구분하며 0으로 조용히 채우지 않습니다.
    """
    df = df.copy()
    if 'Amount' not in df.columns:
        if 'Close' in df.columns and 'Volume' in df.columns:
            df['Amount'] = df['Close'] * df['Volume']
            return df, "FULLY_ESTIMATED"
        return df, "UNESTIMABLE"
            
    amount_null_count = df['Amount'].isnull().sum()
    total_len = len(df)
    
    if amount_null_count == total_len:
        if 'Close' in df.columns and 'Volume' in df.columns:
            df['Amount'] = df['Close'] * df['Volume']
            return df, "FULLY_ESTIMATED"
        return df, "NO_AMOUNT"
    elif amount_null_count > 0:
        if 'Close' in df.columns and 'Volume' in df.columns:
            df['Amount'] = df['Amount'].fillna(df['Close'] * df['Volume'])
        return df, "PARTIALLY_MISSING"
    return df, "ACTUAL"


def process_and_validate_stock_data(df, execution_date=None, mode='realtime_after_close', dedup_policy='exclude'):
    """
    날짜 정규화, 중복 검증, 최신 봉 처리 및 거래대금 검증을 통합 수행합니다.
    """
    if df is None or df.empty:
        return None, None, {"status": "REJECTED", "reason": "빈 데이터셋"}

    # 컬럼명 표준화
    df = normalize_columns(df)

    # 날짜 정규화 및 정렬
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).floor('D')
    df = df.sort_index()

    # 중복 날짜 처리
    dup_mask = df.index.duplicated(keep=False)
    dup_count = dup_mask.sum()
    dup_dates = df.index[dup_mask].unique().tolist()
    
    validation_log = {
        "dup_count": dup_count,
        "dup_dates": [d.strftime('%Y-%m-%d') for d in dup_dates],
        "missing_rows": df.isnull().any(axis=1).sum(),
        "status": "PASSED",
        "reason": ""
    }

    if dup_count > 0:
        if dedup_policy == 'exclude':
            validation_log["status"] = "REJECTED"
            validation_log["reason"] = f"중복 날짜 {len(dup_dates)}개 발견으로 제외 정책 적용"
            return None, None, validation_log
        elif dedup_policy == 'keep_last':
            df = df[~df.index.duplicated(keep='last')].copy()

    # execution_date 기준 미래 데이터 제거
    if execution_date is not None:
        exec_date = pd.Timestamp(execution_date).tz_localize(None).floor('D')
        df = df[df.index <= exec_date].copy()
        
    if df.empty:
        validation_log["status"] = "REJECTED"
        validation_log["reason"] = "기준일 이전 데이터 없음"
        return None, None, validation_log

    last_date = df.index[-1]

    # 모드별 최신 봉 처리
    if mode == 'realtime_intraday':
        if execution_date and last_date == pd.Timestamp(execution_date).tz_localize(None).floor('D'):
            if len(df) > 1:
                analysis_date = df.index[-2]
                df_analysis = df.iloc[:-1].copy()
            else:
                validation_log["status"] = "REJECTED"
                validation_log["reason"] = "장중 봉 제외 후 데이터 부족"
                return None, None, validation_log
        else:
            analysis_date = last_date
            df_analysis = df.copy()
    else:
        analysis_date = last_date
        df_analysis = df.copy()

    # 거래대금 처리
    df_analysis, amount_status = validate_and_process_amount(df_analysis)
    validation_log["amount_status"] = amount_status

    return df_analysis, analysis_date, validation_log


# =====================================================================
# 3. 박스 채널 및 특징 생성
# =====================================================================
def calculate_box_channel(df, window=20):
    """
    박스권 지표를 계산하고 초반 결측 구간은 NaN으로 유지합니다.
    """
    df = df.copy()
    
    df['Prev_Box_High'] = df['High'].shift(1).rolling(window=window).max()
    df['Prev_Box_Low'] = df['Low'].shift(1).rolling(window=window).min()
    df['Prev_Box_Width'] = df['Prev_Box_High'] - df['Prev_Box_Low']
    
    df['Curr_Box_High'] = df['High'].rolling(window=window).max()
    df['Curr_Box_Low'] = df['Low'].rolling(window=window).min()
    df['Curr_Box_Width'] = df['Curr_Box_High'] - df['Curr_Box_Low']
    
    df['Box_Position'] = (df['Close'] - df['Prev_Box_Low']) / (df['Prev_Box_Width'].replace(0, np.nan))
    df['Box_Squeeze_Ratio'] = df['Curr_Box_Width'] / (df['Prev_Box_Width'].replace(0, np.nan))
    
    df['Intraday_Breakout_Top'] = df['High'] > df['Prev_Box_High']
    df['Close_Breakout_Top']    = df['Close'] > df['Prev_Box_High']
    df['Fail_Breakout_Top']     = df['Intraday_Breakout_Top'] & (~df['Close_Breakout_Top'])
    
    df['Intraday_Breakdown_Low'] = df['Low'] < df['Prev_Box_Low']
    df['Close_Breakdown_Low']    = df['Close'] < df['Prev_Box_Low']
    df['Recovered_From_Low']     = df['Intraday_Breakdown_Low'] & (df['Close'] >= df['Prev_Box_Low'])
    
    return df


def add_features(df, cfg=None):
    """
    이동평균, 등락률, 박스권 지표 등을 생성합니다.
    """
    df = normalize_columns(df)
    df = calculate_box_channel(df, window=cfg.get('box_window', 20) if cfg else 20)
    
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Pct_Change'] = df['Close'].pct_change()
    
    return df


# =====================================================================
# 4. 점수 측정 통합 표준 함수 (score_latest)
# =====================================================================
def score_latest(df, market_name="KOSPI", cfg=None):
    """
    통합 표준 사양을 준수하는 스코어링 함수입니다.
    반환값: Dict (valid, score, reasons, penalties, missing, market)
    """
    if cfg is None:
        cfg = {}

    default_core_features = ['Close', 'Volume', 'Amount', 'MA5', 'MA20', 'Prev_Box_High', 'Prev_Box_Low']
    core_features = cfg.get('core_features', default_core_features)
    
    result = {
        "valid": True,
        "score": 50.0,
        "reasons": [],
        "penalties": [],
        "missing": [],
        "market": market_name
    }

    if df is None or df.empty or len(df) < 20:
        result["valid"] = False
        result["reasons"].append("데이터 길이 20일 미만으로 분석 불가")
        return result

    latest = df.iloc[-1]

    # A. 핵심 결측 특징 검증
    for feat in core_features:
        if feat not in latest or pd.isnull(latest[feat]):
            result["missing"].append(feat)
    
    if result["missing"]:
        result["valid"] = False
        result["penalties"].append(f"핵심 특징 결측 ({', '.join(result['missing'])})")
        return result

    # B. 최소 거래대금 필터 (예: min_amount 단위 원/천원 등 cfg 설정에 따름)
    min_amount = cfg.get('min_amount', 1_000_000_000) # 기본 10억원
    if latest['Amount'] < min_amount:
        result["valid"] = False
        result["penalties"].append(f"최소 거래대금 미달 (현재: {latest['Amount']:,.0f} < 기준: {min_amount:,.0f})")

    # C. 오늘 상승 제외 규칙
    exclude_today_rise = cfg.get('exclude_today_rise', False)
    if exclude_today_rise and latest.get('Pct_Change', 0) > 0:
        result["valid"] = False
        result["penalties"].append(f"오늘 상승 종목 제외 규칙 적용 (당일 등락률: {latest['Pct_Change']*100:.2f}%)")

    # D. 박스권 기반 점수 산출
    score = 50.0
    if latest.get('Close_Breakout_Top', False):
        score += 25.0
        result["reasons"].append("이전 20일 박스권 상단 종가 돌파(+25)")
    elif latest.get('Fail_Breakout_Top', False):
        score -= 15.0
        result["reasons"].append("박스 상단 장중 돌파 후 종가 밀림(-15)")

    if latest.get('Recovered_From_Low', False):
        score += 15.0
        result["reasons"].append("박스 하단 장중 이탈 후 종가 회복(+15)")
    elif latest.get('Close_Breakdown_Low', False):
        score -= 25.0
        result["reasons"].append("이전 20일 박스권 하단 종가 이탈(-25)")

    squeeze = latest.get('Box_Squeeze_Ratio', np.nan)
    if pd.notnull(squeeze) and squeeze < 0.8:
        score += 10.0
        result["reasons"].append("박스권 폭 응축 수렴(+10)")

    result["score"] = max(0.0, min(100.0, score))
    return result


# =====================================================================
# 5. 종목 분석 파이프라인 (process_stock) 및 후보 화면 지원
# =====================================================================
def process_stock(df, symbol="005930", market_name="KOSPI", cfg=None, execution_date=None, mode='realtime_after_close'):
    """
    단일 종목을 검증, 지표 생성, 점수 측정까지 완료하여 통합 표준 결과를 반환합니다.
    """
    if cfg is None:
        cfg = {}

    df_validated, analysis_date, val_log = process_and_validate_stock_data(
        df, execution_date=execution_date, mode=mode, dedup_policy=cfg.get('dedup_policy', 'exclude')
    )

    if df_validated is None:
        return {
            "symbol": symbol,
            "market": market_name,
            "analysis_date": None,
            "valid": False,
            "score": 0.0,
            "reasons": [],
            "penalties": [val_log["reason"]],
            "missing": [],
            "validation_log": val_log
        }

    df_featured = add_features(df_validated, cfg=cfg)
    score_res = score_latest(df_featured, market_name=market_name, cfg=cfg)
    
    # 공통 출력 딕셔너리 구성
    output = {
        "symbol": symbol,
        "market": market_name,
        "analysis_date": analysis_date.strftime('%Y-%m-%d') if analysis_date else None,
        "valid": score_res["valid"],
        "score": score_res["score"],
        "reasons": score_res["reasons"],
        "penalties": score_res["penalties"],
        "missing": score_res["missing"],
        "validation_log": val_log,
        "latest_close": df_featured.iloc[-1]['Close'] if not df_featured.empty else None
    }
    return output


# =====================================================================
# 6. 표준 백테스트 실행 함수 (run_backtest_single_date)
# =====================================================================
def run_backtest_single_date(df, t_date, trading_days_dict, market_name="KOSPI", cfg=None):
    """
    t일 시점 데이터만으로 추천을 생성하고, t+1일 시가 진입 / 종가 청산 성과를 측정합니다.
    """
    if cfg is None:
        cfg = {}

    t_date = pd.Timestamp(t_date).tz_localize(None).floor('D')
    
    # 1. t일까지의 데이터만 슬라이싱 (미래 데이터 차단)
    df_t = df[df.index <= t_date].copy()
    if df_t.empty or df_t.index[-1] != t_date:
        return None, f"기준일 t({t_date.strftime('%Y-%m-%d')}) 데이터 부재"

    # 2. t+1 거래일 조회 (없으면 즉시 사유 기록 후 중단)
    next_t, status_msg = get_next_trading_day(t_date, trading_days_dict)
    if next_t is None:
        return None, f"t+1 거래일 조회 실패: {status_msg}"

    if next_t not in df.index:
        return None, f"원본 데이터셋에 t+1 거래일({next_t.strftime('%Y-%m-%d')}) 데이터가 존재하지 않음"

    t_plus_1_row = df.loc[next_t]

    # 3. t일 특징 생성 및 점수 산출
    df_t_featured = add_features(df_t, cfg=cfg)
    score_res = score_latest(df_t_featured, market_name=market_name, cfg=cfg)

    if not score_res["valid"]:
        return None, f"t일 기준 매수 추천 조건 미달 ({', '.join(score_res['penalties'])})"

    # 4. t+1 성과 계산 (시가 진입 / 종가 청산 vs 종가 간)
    p_t_close = df_t_featured.iloc[-1]['Close']
    p_t1_open = t_plus_1_row['Open']
    p_t1_close = t_plus_1_row['Close']

    # 세전(Raw) 수익률
    ret_open_to_close_raw = (p_t1_close - p_t1_open) / p_t1_open
    ret_close_to_close_raw = (p_t1_close - p_t_close) / p_t_close

    # 수수료, 세금, 슬리피지 차감 (기본 설정: 수수료+세금 0.2%, 슬리피지 0.1%)
    fee_rate = cfg.get('fee_rate', 0.002)
    slippage = cfg.get('slippage', 0.001)
    total_cost = fee_rate + slippage

    ret_open_to_close_net = ret_open_to_close_raw - total_cost
    ret_close_to_close_net = ret_close_to_close_raw - total_cost

    backtest_result = {
        "t_date": t_date.strftime('%Y-%m-%d'),
        "t_plus_1_date": next_t.strftime('%Y-%m-%d'),
        "score": score_res["score"],
        "reasons": score_res["reasons"],
        "p_t_close": p_t_close,
        "p_t1_open": p_t1_open,
        "p_t1_close": p_t1_close,
        "ret_open_to_close_raw": ret_open_to_close_raw,
        "ret_open_to_close_net": ret_open_to_close_net,
        "ret_close_to_close_raw": ret_close_to_close_raw,
        "ret_close_to_close_net": ret_close_to_close_net,
    }

    return backtest_result, "성공"
