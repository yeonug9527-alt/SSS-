import datetime as dt
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# V10.0 — 국내주식 급등 전조 / 다음 거래일 후보 스캐너
# ============================================================
st.set_page_config(
    page_title="V10 국내주식 급등 전조 스캐너",
    page_icon="📈",
    layout="wide",
)

st.title("📈 V10 국내주식 급등 전조 스캐너")
st.caption(
    "목표: 이미 급등한 종목을 추격하기보다, 박스권·변동성 축소·거래대금·상대강도 "
    "등을 바탕으로 다음 거래일 후보를 탐색합니다."
)

KST = dt.timezone(dt.timedelta(hours=9))
NOW_KST = dt.datetime.now(KST)
TODAY = NOW_KST.date()


# ============================================================
# 설정
# ============================================================
DEFAULT_CONFIG = {
    "min_trading_value": 5_000_000_000,  # 50억원
    "max_today_return": 3.0,             # 기본: +3% 초과 제외
    "min_history": 80,
    "lookback_days": 180,
    "top_n": 30,
    "workers": 8,
    "missing_core_action": "제외",
    "fee_rate": 0.00015,
    "tax_rate": 0.0020,
    "slippage_rate": 0.0005,
}

if "config" not in st.session_state:
    st.session_state.config = DEFAULT_CONFIG.copy()

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if "invalid_results" not in st.session_state:
    st.session_state.invalid_results = None


# ============================================================
# 유틸 및 거래일 계산
# ============================================================
def fmt_pct(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x:.{digits}f}%"


def safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def normalize_columns(df):
    mapping = {
        "Date": "Date",
        "date": "Date",
        "날짜": "Date",
        "Open": "Open",
        "open": "Open",
        "시가": "Open",
        "High": "High",
        "high": "High",
        "고가": "High",
        "Low": "Low",
        "low": "Low",
        "저가": "Low",
        "Close": "Close",
        "close": "Close",
        "종가": "Close",
        "Volume": "Volume",
        "volume": "Volume",
        "거래량": "Volume",
        "Amount": "Amount",
        "amount": "Amount",
        "거래대금": "Amount",
        "Code": "Code",
        "code": "Code",
        "종목코드": "Code",
        "Name": "Name",
        "name": "Name",
        "종목명": "Name",
        "Market": "Market",
        "market": "Market",
        "시장": "Market",
        "Sector": "Sector",
        "sector": "Sector",
        "업종": "Sector",
    }
    return df.rename(columns={c: mapping.get(c, c) for c in df.columns})


@st.cache_data(ttl=86400, show_spinner=False)
def get_krx_trading_days(year):
    """지정된 연도의 실제 한국 주식시장 거래일 목록을 반환합니다."""
    # 연도별 주요 음력 명절 및 대체공휴일 정의
    lunar_holidays = {
        2025: [
            dt.date(2025, 1, 28), dt.date(2025, 1, 29), dt.date(2025, 1, 30), # 설날
            dt.date(2025, 3, 3),   # 삼일절 대체
            dt.date(2025, 5, 5),   # 어린이날/부처님오신날
            dt.date(2025, 5, 6),   # 대체휴일
            dt.date(2025, 10, 6), dt.date(2025, 10, 7), dt.date(2025, 10, 8), # 추석
        ],
        2026: [
            dt.date(2026, 2, 16), dt.date(2026, 2, 17), dt.date(2026, 2, 18), # 설날
            dt.date(2026, 3, 2),   # 삼일절 대체
            dt.date(2026, 5, 5),   # 어린이날
            dt.date(2026, 5, 25),  # 부처님오신날 대체
            dt.date(2026, 9, 24), dt.date(2026, 9, 25), dt.date(2026, 9, 28), # 추석 및 대체
        ],
        2027: [
            dt.date(2027, 2, 8), dt.date(2027, 2, 9), dt.date(2027, 2, 10), # 설날
            dt.date(2027, 5, 13),  # 부처님오신날
            dt.date(2027, 9, 14), dt.date(2027, 9, 15), dt.date(2027, 9, 16), # 추석
        ]
    }

    # 양력 고정 휴장일
    fixed_holidays = {
        dt.date(year, 1, 1),   # 신정
        dt.date(year, 3, 1),   # 삼일절
        dt.date(year, 5, 1),   # 근로자의 날
        dt.date(year, 5, 5),   # 어린이날
        dt.date(year, 6, 6),   # 현충일
        dt.date(year, 8, 15),  # 광복절
        dt.date(year, 10, 3),  # 개천절
        dt.date(year, 10, 9),  # 한글날
        dt.date(year, 12, 25), # 성탄절
        dt.date(year, 12, 31), # 연말 휴장일
    }

    all_holidays = fixed_holidays.union(set(lunar_holidays.get(year, [])))

    start = dt.date(year, 1, 1)
    end = dt.date(year, 12, 31)
    all_days = [start + dt.timedelta(days=x) for x in range((end - start).days + 1)]
    return [d for d in all_days if d.weekday() < 5 and d not in all_holidays]


def get_next_trading_day(base_date):
    """실제 거래일 목록을 바탕으로 다음 거래일을 반환합니다 (BDay 완전 대체)."""
    trading_days = get_krx_trading_days(base_date.year)
    if base_date >= trading_days[-1]:
        trading_days += get_krx_trading_days(base_date.year + 1)
    for td in trading_days:
        if td > base_date:
            return td
    return base_date


# ============================================================
# KRX 목록
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_listing():
    df = fdr.StockListing("KRX").copy()
    df = normalize_columns(df)

    code_col = next((c for c in ["Code", "Symbol"] if c in df.columns), None)
    name_col = next((c for c in ["Name", "종목명"] if c in df.columns), None)

    if code_col is None or name_col is None:
        raise ValueError("KRX 종목 목록에서 종목코드/종목명을 찾지 못했습니다.")

    df["Code"] = df[code_col].astype(str).str.zfill(6)
    df["Name"] = df[name_col].astype(str)

    text = (
        df["Name"].fillna("")
        + " "
        + df.get("Market", pd.Series("", index=df.index)).fillna("").astype(str)
    )

    exclude_pattern = (
        r"ETF|ETN|리츠|REIT|스팩|SPAC|인버스|레버리지|"
        r"선물|우선주|우B|우선|상장지수"
    )
    mask = ~text.str.contains(exclude_pattern, case=False, regex=True, na=False)
    df = df.loc[mask].copy()

    if "Market" in df.columns:
        df = df[df["Market"].astype(str).str.upper().isin(["KOSPI", "KOSDAQ"])].copy()

    return df[["Code", "Name"] + ([c for c in ["Market", "Sector"] if c in df.columns])].drop_duplicates("Code")


# ============================================================
# 일봉 데이터
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_daily_data(code, days=180):
    end = dt.datetime.now(KST).date()
    start = end - dt.timedelta(days=days)

    df = fdr.DataReader(code, start, end).copy()
    if df.empty:
        return pd.DataFrame()

    df = normalize_columns(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Amount" not in df.columns:
        df["Amount"] = df["Close"] * df["Volume"]

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ============================================================
# 데이터 품질
# ============================================================
def validate_daily_data(df, cfg):
    issues = []

    if df.empty:
        return False, ["데이터 없음"]

    if df["Date"].duplicated().any():
        issues.append("중복 거래일")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Amount"]
    for c in numeric_cols:
        if c not in df.columns:
            issues.append(f"{c} 컬럼 없음")
        elif df[c].isna().all():
            issues.append(f"{c} 전체 결측")

    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        bad_price = (
            (df["Open"] <= 0)
            | (df["High"] <= 0)
            | (df["Low"] <= 0)
            | (df["Close"] <= 0)
            | (df["High"] < df["Low"])
        )
        if bad_price.any():
            issues.append("가격 이상값")

    if "Volume" in df.columns and (df["Volume"] < 0).any():
        issues.append("거래량 음수")

    if "Amount" in df.columns and (df["Amount"] < 0).any():
        issues.append("거래대금 음수")

    future = pd.to_datetime(df["Date"]).dt.date > TODAY
    if future.any():
        issues.append("미래 날짜 데이터")

    enough = len(df) >= cfg["min_history"]
    if not enough:
        issues.append(f"데이터 부족({len(df)}일)")

    return len(issues) == 0, issues


# ============================================================
# 특징 생성
# ============================================================
def add_features(df):
    x = df.copy().sort_values("Date").reset_index(drop=True)

    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]
    amount = x["Amount"]

    x["ret_1"] = close.pct_change(1) * 100
    x["ret_3"] = close.pct_change(3) * 100
    x["ret_5"] = close.pct_change(5) * 100
    x["ret_10"] = close.pct_change(10) * 100
    x["ret_20"] = close.pct_change(20) * 100

    x["ma20"] = close.rolling(20).mean()
    x["ma60"] = close.rolling(60).mean()
    x["disp20"] = (close / x["ma20"] - 1) * 100
    x["disp60"] = (close / x["ma60"] - 1) * 100

    x["high20"] = high.rolling(20).max()
    x["low20"] = low.rolling(20).min()
    x["drawdown20"] = (close / x["high20"] - 1) * 100
    x["rise_from_low20"] = (close / x["low20"] - 1) * 100

    x["up_ratio_20"] = (x["ret_1"] > 0).rolling(20).mean() * 100

    direction = np.sign(x["ret_1"].fillna(0))
    groups = (direction != direction.shift()).cumsum()
    streak = direction.groupby(groups).cumcount() + 1
    x["streak"] = streak * direction

    x["gap"] = (x["Open"] / x["Close"].shift(1) - 1) * 100

    denom = (high - low).replace(0, np.nan)
    x["close_position"] = (close - low) / denom * 100

    x["vol_ma20"] = volume.rolling(20).mean()
    x["vol_ma60"] = volume.rolling(60).mean()
    x["amount_ma20"] = amount.rolling(20).mean()
    x["amount_ma60"] = amount.rolling(60).mean()

    x["vol_ratio20"] = volume / x["vol_ma20"]
    x["vol_5_vs_60"] = volume.rolling(5).mean() / x["vol_ma60"]
    x["amount_ratio20"] = amount / x["amount_ma20"]
    x["amount_5_change"] = amount.rolling(5).mean().pct_change(5) * 100

    x["volume_spike"] = x["vol_ratio20"] >= 2.0
    x["distribution_day"] = (
        (x["vol_ratio20"] >= 1.5) & (x["close_position"] < 50)
    )

    x["range_pct"] = (high - low) / low * 100
    x["volatility5"] = x["ret_1"].rolling(5).std()
    x["volatility20"] = x["ret_1"].rolling(20).std()

    for n in [5, 10, 20]:
        bh = high.rolling(n).max()
        bl = low.rolling(n).min()
        width = (bh - bl) / bl.replace(0, np.nan)
        position = (close - bl) / (bh - bl).replace(0, np.nan)

        x[f"box_high_{n}"] = bh
        x[f"box_low_{n}"] = bl
        x[f"box_width_{n}"] = width * 100
        x[f"box_position_{n}"] = position * 100

        near_top = close >= bh * 0.97
        x[f"box_top_touch_{n}"] = near_top.rolling(5).sum()

        below_low = close < bl.shift(1)
        x[f"box_bottom_break_{n}"] = below_low.rolling(5).sum()

    x["box5_vs_box20"] = x["box_width_5"] / x["box_width_20"]

    # 기존 x["low"], x["high"] 소문자 참조 오류 수정 -> low, high 변수 활용
    x["low5_slope"] = low.rolling(5).apply(
        lambda z: np.polyfit(np.arange(len(z)), z, 1)[0]
        if np.isfinite(z).all() else np.nan,
        raw=True,
    )
    x["high5_slope"] = high.rolling(5).apply(
        lambda z: np.polyfit(np.arange(len(z)), z, 1)[0]
        if np.isfinite(z).all() else np.nan,
        raw=True,
    )

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = 100 - (100 / (1 + rs))

    return x


# ============================================================
# 점수
# ============================================================
def score_latest(x, market_name, cfg):
    if x.empty:
        return None

    r = x.iloc[-1]

    core = [
        "ret_1", "ret_5", "ret_20",
        "box_width_20", "box_position_20",
        "vol_ratio20", "amount_ratio20",
        "volatility5", "volatility20",
    ]
    missing_core = [c for c in core if pd.isna(r.get(c, np.nan))]

    if missing_core and cfg["missing_core_action"] == "제외":
        return {
            "valid": False,
            "reason": "핵심 특징 결측: " + ", ".join(missing_core),
        }

    liquidity_ok = (
        pd.notna(r.get("Amount"))
        and r["Amount"] >= cfg["min_trading_value"]
    )
    if not liquidity_ok:
        return {"valid": False, "reason": "거래대금 부족"}

    today_ret = r.get("ret_1", np.nan)

    if pd.notna(today_ret) and today_ret > cfg["max_today_return"]:
        return {
            "valid": False,
            "reason": f"오늘 수익률 {today_ret:.2f}% > 제외기준 {cfg['max_today_return']:.2f}%",
        }

    score = 0.0
    reasons = []
    penalties = []

    if pd.notna(r.get("ret_5")) and r["ret_5"] > 0:
        score += 2
        reasons.append("최근 5일 상승 흐름")

    if pd.notna(r.get("amount_5_change")) and r["amount_5_change"] > 0:
        score += 2
        reasons.append("최근 거래대금 증가")

    if pd.notna(r.get("vol_ratio20")) and r["vol_ratio20"] > 1:
        score += 2
        reasons.append("전일 거래량이 20일 평균 이상")

    if pd.notna(r.get("box_position_20")) and r["box_position_20"] >= 80:
        score += 2
        reasons.append("20일 박스 상단 부근")

    if (
        pd.notna(r.get("box5_vs_box20"))
        and r["box5_vs_box20"] < 0.75
    ):
        score += 2
        reasons.append("최근 5일 박스 폭이 20일보다 좁음")

    if pd.notna(r.get("low5_slope")) and r["low5_slope"] > 0:
        score += 2
        reasons.append("최근 저점 상승")

    if bool(r.get("distribution_day", False)):
        score -= 3
        penalties.append("거래량 급증 후 종가가 고가에서 크게 밀림")

    if pd.notna(r.get("box_top_touch_20")) and r["box_top_touch_20"] >= 3:
        score -= 2
        penalties.append("박스 상단 반복 테스트")

    if pd.notna(r.get("ret_5")) and r["ret_5"] >= 10:
        score -= 2
        penalties.append("최근 5일 상승률 과도")

    if pd.notna(r.get("disp20")) and r["disp20"] >= 12:
        score -= 3
        penalties.append("20일선 대비 과도한 이격")

    return {
        "valid": True,
        "score": score,
        "today_return": today_ret,
        "ret_5": r.get("ret_5"),
        "ret_20": r.get("ret_20"),
        "box_width_20": r.get("box_width_20"),
        "box_position_20": r.get("box_position_20"),
        "vol_ratio20": r.get("vol_ratio20"),
        "amount_5_change": r.get("amount_5_change"),
        "disp20": r.get("disp20"),
        "rsi14": r.get("rsi14"),
        "reasons": reasons,
        "penalties": penalties,
        "missing": ", ".join(missing_core) if missing_core else "",
        "market": market_name,
    }


def process_stock(row, cfg):
    code = row["Code"]
    name = row["Name"]
    market = row.get("Market", "미상")

    try:
        df = get_daily_data(code, cfg["lookback_days"])
        valid, issues = validate_daily_data(df, cfg)

        if not valid:
            return {
                "Code": code,
                "Name": name,
                "Market": market,
                "valid": False,
                "reason": "; ".join(issues),
            }

        actual_analysis_date = df["Date"].iloc[-1].date()
        target_date = get_next_trading_day(actual_analysis_date)

        feat = add_features(df)
        result = score_latest(feat, market, cfg)

        if result is None:
            return {"Code": code, "Name": name, "valid": False, "reason": "분석 실패"}

        result.update({
            "Code": code,
            "Name": name,
            "Market": market,
            "analysis_date": str(actual_analysis_date),
            "target_date": str(target_date),
        })
        return result

    except Exception as e:
        return {
            "Code": code,
            "Name": name,
            "Market": market,
            "valid": False,
            "reason": f"{type(e).__name__}: {e}",
        }


# ============================================================
# UI
# ============================================================
with st.sidebar:
    st.header("⚙️ 분석 설정")

    market_choice = st.multiselect(
        "시장",
        ["KOSPI", "KOSDAQ"],
        default=["KOSPI", "KOSDAQ"],
    )

    min_amount_eok = st.number_input(
        "최소 거래대금 (억원)",
        min_value=1.0,
        value=50.0,
        step=10.0,
    )

    exclude_rule = st.selectbox(
        "오늘 상승 종목 제외 기준",
        [
            "오늘 +0% 초과 제외",
            "오늘 +3% 초과 제외",
            "오늘 +5% 초과 제외",
            "오늘 +10% 초과 제외",
        ],
        index=1,
    )

    top_n = st.number_input(
        "상위 추천 종목 수",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
    )

    missing_action = st.selectbox(
        "핵심 특징 결측 처리",
        ["제외", "결측 허용 후 점수 미반영"],
        index=0,
    )

    st.divider()
    st.subheader("비용 설정")

    fee = st.number_input(
        "수수료율 (%)",
        min_value=0.0,
        value=0.015,
        step=0.005,
        format="%.3f",
    )
    tax = st.number_input(
        "세금율 (%)",
        min_value=0.0,
        value=0.20,
        step=0.05,
        format="%.3f",
    )
    slippage = st.number_input(
        "슬리피지 (%)",
        min_value=0.0,
        value=0.05,
        step=0.01,
        format="%.3f",
    )

    st.session_state.config.update({
        "min_trading_value": int(min_amount_eok * 100_000_000),
        "top_n": int(top_n),
        "missing_core_action": "제외" if missing_action == "제외" else "허용",
        "fee_rate": fee / 100,
        "tax_rate": tax / 100,
        "slippage_rate": slippage / 100,
    })

    if exclude_rule.startswith("오늘 +0"):
        st.session_state.config["max_today_return"] = 0.0
    elif exclude_rule.startswith("오늘 +3"):
        st.session_state.config["max_today_return"] = 3.0
    elif exclude_rule.startswith("오늘 +5"):
        st.session_state.config["max_today_return"] = 5.0
    elif exclude_rule.startswith("오늘 +10"):
        st.session_state.config["max_today_return"] = 10.0


# ============================================================
# 시간/기준일 표시
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_actual_market_date():
    test_df = fdr.DataReader('069500', NOW_KST.date() - dt.timedelta(days=10))
    if not test_df.empty:
        return test_df.index[-1].date()
    return NOW_KST.date()

ACTUAL_MARKET_DATE = get_actual_market_date()
NEXT_TRADING_DATE = get_next_trading_day(ACTUAL_MARKET_DATE)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🕐 프로그램 실행", NOW_KST.strftime("%Y-%m-%d %H:%M:%S"))
c2.metric("📅 프로그램 실행일", NOW_KST.strftime("%Y-%m-%d"))
c3.metric("📊 실제 분석 기준일", ACTUAL_MARKET_DATE.strftime("%Y-%m-%d"))
c4.metric("🎯 다음 거래일(예측)", NEXT_TRADING_DATE.strftime("%Y-%m-%d"))

st.info(
    "💡 프로그램 실행일과 시장의 실제 데이터 기준일을 분리하여 표시합니다. "
    "다음 거래일은 한국거래소(KRX) 실제 휴장일을 반영하여 산출됩니다."
)


tabs = st.tabs([
    "🚀 다음 거래일 후보",
    "🧪 전략 비교",
    "🛡️ 데이터 검증",
    "ℹ️ 설계 정보",
])


# ============================================================
# 후보 스캔
# ============================================================
with tabs[0]:
    st.subheader("🚀 급등 전조 후보")

    st.markdown(
        f"""
**현재 설정**
- 최소 거래대금: **{min_amount_eok:.0f}억원**
- 오늘 상승 제외: **+{st.session_state.config["max_today_return"]:.0f}% 초과**
- 분석 기준일: **{ACTUAL_MARKET_DATE.isoformat()}**
- 예측 대상일: **{NEXT_TRADING_DATE.isoformat()}**
"""
    )

    if not market_choice:
        st.warning("최소 한 개 시장을 선택해주세요.")
    else:
        if st.button("🚀 고속 스캔 시작", type="primary", use_container_width=True):
            listing = get_listing()

            if "Market" in listing.columns:
                listing = listing[listing["Market"].isin(market_choice)].copy()

            st.write(f"대상 종목: **{len(listing):,}개**")

            progress = st.progress(0)
            status = st.empty()

            results = []
            rows = listing.to_dict("records")

            workers = st.session_state.config["workers"]

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_stock, row, st.session_state.config): row["Code"]
                    for row in rows
                }

                total = len(futures)
                for i, future in enumerate(as_completed(futures), start=1):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append({
                            "Code": futures[future],
                            "valid": False,
                            "reason": str(e),
                        })

                    progress.progress(i / max(total, 1))
                    status.text(f"분석 중... {i:,} / {total:,}")

            st.session_state.scan_results = [r for r in results if r.get("valid")]
            st.session_state.invalid_results = [r for r in results if not r.get("valid")]

        # 결과가 세션 상태에 존재하는 경우 출력
        if st.session_state.scan_results is not None:
            valid_results = st.session_state.scan_results
            invalid_results = st.session_state.invalid_results

            if not valid_results:
                st.warning("조건을 만족하는 후보가 없습니다.")
                st.caption(f"제외/오류 종목: {len(invalid_results):,}")
            else:
                result_df = pd.DataFrame(valid_results)
                result_df = result_df.sort_values(
                    ["score", "box_position_20"],
                    ascending=[False, False],
                ).head(int(top_n))

                result_df["추천 근거"] = result_df.apply(
                    lambda r: " / ".join(r["reasons"]) if r["reasons"] else "특별한 가점 없음",
                    axis=1,
                )
                result_df["감점 요인"] = result_df.apply(
                    lambda r: " / ".join(r["penalties"]) if r["penalties"] else "-",
                    axis=1,
                )

                display_cols = [
                    "Code", "Name", "Market", "score",
                    "today_return", "ret_5", "ret_20",
                    "box_width_20", "box_position_20",
                    "vol_ratio20", "amount_5_change",
                    "rsi14", "추천 근거", "감점 요인", "missing",
                ]

                display_df = result_df[display_cols].copy()
                display_df.columns = [
                    "종목코드", "종목명", "시장", "점수",
                    "오늘 수익률", "5일 수익률", "20일 수익률",
                    "20일 박스폭", "박스 위치",
                    "거래량/20일평균", "5일 거래대금 변화",
                    "RSI", "추천 근거", "감점 요인", "결측",
                ]

                st.success(f"추천 후보 {len(display_df)}개")
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                csv = display_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "📥 결과 CSV 저장",
                    data=csv,
                    file_name=f"v10_candidates_{ACTUAL_MARKET_DATE.isoformat()}.csv",
                    mime="text/csv",
                )

                st.subheader("⚠️ 제외/오류 요약")
                if invalid_results:
                    invalid_df = pd.DataFrame(invalid_results)
                    if "reason" in invalid_df.columns:
                        reason_counts = (
                            invalid_df["reason"]
                            .fillna("알 수 없음")
                            .value_counts()
                            .head(20)
                            .rename_axis("사유")
                            .reset_index(name="종목 수")
                        )
                        st.dataframe(reason_counts, use_container_width=True, hide_index=True)


# ============================================================
# 전략 비교
# ============================================================
with tabs[1]:
    st.subheader("🧪 전략 비교 설계")
    st.info(
        "현재 단계에서는 과거 데이터 전체를 자동 다운로드하여 장기간 백테스트하는 기능은 "
        "아직 분리하지 않았습니다. 다음 단계에서 동일한 특징 생성기를 사용해 시간순 백테스트를 추가합니다."
    )

    strategy_df = pd.DataFrame({
        "전략": [
            "A. 전체 종목",
            "B. 박스권",
            "C. 박스권 + 거래량",
            "D. 박스권 + 거래량 + 상대강도",
            "E. D + 오늘 상승 제외",
            "F. E + 수급",
            "G. 오늘 +15% 급등 후속",
            "H. 최종 급등 전조 전략",
        ],
        "현재 상태": [
            "준비",
            "준비",
            "준비",
            "시장지수 연결 후",
            "현재 후보 필터에 일부 적용",
            "수급 데이터 연결 후",
            "별도 연구 필요",
            "현재 V10 핵심",
        ],
    })
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)


# ============================================================
# 데이터 검증
# ============================================================
with tabs[2]:
    st.subheader("🛡️ 데이터 품질 검증")

    st.write(
        "현재 스캔 시 각 종목별로 중복 날짜, 가격 이상값, 음수 거래량/거래대금, "
        "미래 날짜, 데이터 부족 여부를 확인합니다."
    )

    if st.button("🔍 샘플 데이터 검증"):
        listing = get_listing()
        sample = listing.head(20).copy()

        rows = []
        for row in sample.to_dict("records"):
            try:
                df = get_daily_data(row["Code"], st.session_state.config["lookback_days"])
                valid, issues = validate_daily_data(df, st.session_state.config)
                rows.append({
                    "종목코드": row["Code"],
                    "종목명": row["Name"],
                    "행 수": len(df),
                    "정상": valid,
                    "문제": "; ".join(issues) if issues else "-",
                    "최종 날짜": str(df["Date"].max().date()) if not df.empty else "-",
                })
            except Exception as e:
                rows.append({
                    "종목코드": row["Code"],
                    "종목명": row["Name"],
                    "행 수": 0,
                    "정상": False,
                    "문제": str(e),
                    "최종 날짜": "-",
                })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# 설계 정보
# ============================================================
with tabs[3]:
    st.subheader("ℹ️ V10 설계 원칙")

    st.markdown(
        """
### 핵심 원칙

1. **오늘 이미 급등한 종목을 추격하지 않는다.**
2. **박스권 → 변동성 축소 → 거래대금 증가 → 상대강도 개선** 구조를 찾는다.
3. 추천 점수와 실제 상승확률을 동일한 의미로 사용하지 않는다.
4. 과거 검증에서는 미래 데이터를 사용하지 않는다.
5. 무작위 train/test가 아니라 시간순 검증을 사용한다.
6. 수급 데이터가 없으면 0으로 대체하지 않는다.
7. 결측값을 무조건 0으로 바꾸지 않는다.
8. 실행 시각, 분석 기준일, 예측 대상일을 별도로 기록한다.
9. 전략의 성과는 비용 차감 전후를 모두 비교한다.
10. 머신러닝은 규칙 기반 버전 검증 후 추가한다.

### 현재 버전의 범위

- 일봉 가격/거래량 기반
- 박스권 특징
- 거래량/거래대금 특징
- 오늘 급등 제외
- 설명 가능한 규칙 점수
- 데이터 품질 검사
- 실행 시각/기준일 표시

### 다음 단계

- 시장지수 상대강도
- 업종 상대강도
- 외국인/기관 수급
- 과거 t+1 라벨 생성
- Walk-forward 백테스트
- 전략 A~H 동일 조건 비교
- 결과 저장 및 재현성 강화
"""
    )
