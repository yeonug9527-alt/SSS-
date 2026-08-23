import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='KRX 종가매매 전략 연구소 V9.8', layout='wide')
st.title('📈 KRX 종가매매 전략 연구소 V9.8')
st.caption('종목 자체의 우수함과 내일 시가 진입 적합성을 분리합니다. 연구용 프로그램입니다.')


def clamp(x, lo=0, hi=100):
    try: return max(lo, min(hi, float(x)))
    except: return lo


def rank_pct(s):
    return s.rank(pct=True, method='average') * 100


def rsi(close, n=14):
    d=close.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n, adjust=False).mean(); ad=dn.ewm(alpha=1/n, adjust=False).mean()
    return (100-100/(1+au/ad.replace(0,np.nan))).fillna(50)


def atr(df,n=14):
    pc=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()


def adx(df,n=14):
    up=df.High.diff(); dn=-df.Low.diff()
    plus=np.where((up>dn)&(up>0),up,0); minus=np.where((dn>up)&(dn>0),dn,0)
    pc=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/n,adjust=False).mean()
    p=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    m=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean().fillna(0)


@st.cache_data(ttl=1800, show_spinner=False)
def listing():
    x=fdr.StockListing('KRX')
    if x is None or x.empty: return pd.DataFrame()
    code=next((c for c in ['Code','Symbol','종목코드'] if c in x),None)
    name=next((c for c in ['Name','종목명'] if c in x),None)
    market=next((c for c in ['Market','시장구분'] if c in x),None)
    if not code or not name: return pd.DataFrame()
    return pd.DataFrame({'Code':x[code].astype(str).str.zfill(6),'Name':x[name].astype(str),'Market':x[market].astype(str) if market else ''}).drop_duplicates('Code')


@st.cache_data(ttl=900, show_spinner=False)
def prices(code,start,end):
    try: x=fdr.DataReader(code,start,end)
    except: return pd.DataFrame()
    if x is None or x.empty or not all(c in x for c in ['Open','High','Low','Close','Volume']): return pd.DataFrame()
    return x[['Open','High','Low','Close','Volume']].replace([np.inf,-np.inf],np.nan).dropna()


def features(df, market):
    if len(df)<100: return None
    c=df.Close; v=df.Volume
    e5=c.ewm(span=5,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e60=c.ewm(span=60,adjust=False).mean()
    R=rsi(c); A=atr(df); AD=adx(df)
    vr=v/v.rolling(20).mean(); ret5=c.pct_change(5)*100; ret20=c.pct_change(20)*100; ret60=c.pct_change(60)*100
    dist=(c/e20-1)*100; break20=c/c.rolling(20).max().shift(1)-1
    mr20=market.Close.pct_change(20).iloc[-1]*100; mr60=market.Close.pct_change(60).iloc[-1]*100
    rel20=ret20.iloc[-1]-mr20; rel60=ret60.iloc[-1]-mr60
    z=df.iloc[-1]
    trend=sum([z.Close>e20.iloc[-1],e5.iloc[-1]>e20.iloc[-1],e20.iloc[-1]>e60.iloc[-1],ret20.iloc[-1]>0])*25
    momentum=.35*clamp((ret5.iloc[-1]+10)/20*100)+.35*clamp((ret20.iloc[-1]+20)/50*100)+.30*clamp((ret60.iloc[-1]+30)/90*100)
    volume=clamp((vr.iloc[-1]-.5)/2.5*100)
    br=90 if break20.iloc[-1]>=0 else 75 if break20.iloc[-1]>-.03 else 55 if break20.iloc[-1]>-.07 else 35
    pull=90 if -4<=dist.iloc[-1]<=2 and e20.iloc[-1]>e60.iloc[-1] else 70 if -8<=dist.iloc[-1]<=4 else 25 if dist.iloc[-1]>12 else 50
    rv=R.iloc[-1]; rs=90 if 50<=rv<=68 else 75 if 45<=rv<50 or 68<rv<=72 else 55 if 35<=rv<45 else 35 if 72<rv<=78 else 20
    ad=clamp((AD.iloc[-1]-10)/30*100); ap=A.iloc[-1]/c.iloc[-1]*100
    atrs=90 if ap<=3 else 75 if ap<=5 else 55 if ap<=7 else 35 if ap<=10 else 15
    ems=90 if 0<=dist.iloc[-1]<=7 else 80 if -3<=dist.iloc[-1]<0 else 60 if dist.iloc[-1]<=12 else 30 if dist.iloc[-1]<=18 else 20
    hot=clamp(max(0,rv-68)*1.5+max(0,dist.iloc[-1]-8)*2+max(0,ap-6)*2)
    avg_value=(c*v).rolling(20).mean().iloc[-1]
    liq=clamp((math.log10(max(avg_value,1))-6)/5*100)
    gaps=(df.Open/df.Close.shift(1)-1).abs().rolling(20).mean().iloc[-1]*100
    gaprisk=clamp(gaps/8*100)
    quality=.16*trend+.14*momentum+.10*volume+.10*br+.08*pull+.07*rs+.07*ad+.05*atrs+.07*ems+.08*clamp((rel20+20)/40*100)+.04*clamp((rel60+30)/80*100)+.04*liq
    entry=.22*pull+.14*rs+.14*ems+.10*br+.10*atrs+.14*(100-hot)+.06*(100-gaprisk)+.10*volume
    final=clamp(.62*quality+.38*entry-.12*hot)
    regime='상승 국면' if market.Close.iloc[-1]>market.Close.ewm(span=20,adjust=False).mean().iloc[-1]>market.Close.ewm(span=60,adjust=False).mean().iloc[-1] and mr20>0 else '하락 국면' if market.Close.iloc[-1]<market.Close.ewm(span=20,adjust=False).mean().iloc[-1]<market.Close.ewm(span=60,adjust=False).mean().iloc[-1] and mr20<0 else '혼조 국면'
    return {'현재가':c.iloc[-1],'종합점수':final,'종목자체점수':clamp(quality),'내일진입점수':clamp(entry),'추세점수':trend,'모멘텀점수':momentum,'거래량점수':volume,'돌파점수':br,'눌림점수':pull,'RSI':rv,'ADX':AD.iloc[-1],'ATR비율':ap,'EMA20이격':dist.iloc[-1],'시장대비20일':rel20,'시장대비60일':rel60,'수익률20일':ret20.iloc[-1],'수익률60일':ret60.iloc[-1],'평균거래대금20일':avg_value,'거래량비율':vr.iloc[-1],'과열위험':hot,'갭위험':gaprisk,'시장국면':regime}


def pattern(r):
    if r.돌파점수>=85 and r.모멘텀점수>=70: return '🚀 돌파 시작형'
    if r.추세점수>=80 and r.모멘텀점수>=70 and r.EMA20이격<=8: return '📈 상승 지속형'
    if r.눌림점수>=82 and r.추세점수>=70: return '🔄 눌림 재상승형'
    if r.추세점수>=65 and r.모멘텀점수>=60: return '📊 상승 추세형'
    return '관찰형'


def reasons(r):
    a=[]; w=[]
    if r.추세점수>=75:a.append('상승 추세가 유지되고 있습니다.')
    if r.시장대비20일>=3:a.append(f'최근 20일 시장보다 {r.시장대비20일:.1f}%p 강합니다.')
    if r.거래량비율>=1.3:a.append(f'거래량이 20일 평균의 약 {r.거래량비율:.1f}배입니다.')
    if r.돌파점수>=80:a.append('최근 고점 돌파 힘이 있습니다.')
    if r.눌림점수>=80:a.append('상승 추세 안에서 진입가격이 비교적 양호합니다.')
    if 50<=r.RSI<=68:a.append('RSI가 상승 중이지만 과열되지 않은 구간입니다.')
    if r.과열위험<25:a.append('단기 과열 위험이 낮습니다.')
    if r.내일진입점수>=75:a.append('내일 시가 진입 적합성이 높습니다.')
    if r.RSI>72:w.append('RSI가 높아 단기 과열을 주의해야 합니다.')
    if r.EMA20이격>10:w.append('20일 평균보다 많이 올라 추격매수 위험이 있습니다.')
    if r.ATR비율>7:w.append('가격 변동성이 높습니다.')
    if r.거래량비율<.7:w.append('거래량이 충분히 붙지 않았습니다.')
    return a,w


def decision(r):
    if r.종합점수>=82 and r.내일진입점수>=75 and r.과열위험<40:return '🟢 적극 관심'
    if r.종합점수>=75 and r.내일진입점수>=65:return '🟢 조건부 추천'
    if r.종합점수>=68:return '🟡 관망'
    return '🔴 제외'


def historical(df):
    if len(df)<150:return dict(유사상황수=0,평균5일=np.nan,평균10일=np.nan,평균20일=np.nan,승률20일=np.nan)
    c=df.Close;e20=c.ewm(span=20,adjust=False).mean();e60=c.ewm(span=60,adjust=False).mean();R=rsi(c);vr=df.Volume/df.Volume.rolling(20).mean();ret20=c.pct_change(20)*100;ap=atr(df)/c*100
    cond=(c>e20)&(e20>e60)&R.between(50,70)&(ret20>0)&(vr>.9)&(ap<8)
    s=pd.DataFrame({'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1,'f20':c.shift(-20)/c-1})[cond].dropna()
    if s.empty:return dict(유사상황수=0,평균5일=np.nan,평균10일=np.nan,평균20일=np.nan,승률20일=np.nan)
    return dict(유사상황수=len(s),평균5일=s.f5.mean()*100,평균10일=s.f10.mean()*100,평균20일=s.f20.mean()*100,승률20일=(s.f20>0).mean()*100)


def analyze(row,start,end,market):
    d=prices(row.Code,start,end)
    if d.empty:return None
    try:
        f=features(d,market)
        if f is None:return None
        f.update({'종목코드':row.Code,'종목명':row.Name,'시장':row.Market})
        f['상승패턴']=pattern(pd.Series(f));f['판단']=decision(pd.Series(f));
        a,w=reasons(pd.Series(f));f['추천이유']=' '.join(a[:5]);f['주의사항']=' '.join(w[:3]);f.update(historical(d))
        return f
    except Exception:return None


def scan(universe,start,end):
    refs=[]
    for code in universe.Code.head(min(20,len(universe))):
        d=prices(code,start,end)
        if not d.empty:refs.append(d.Close.rename(code))
    if refs:
        m=pd.concat(refs,axis=1).pct_change().median(axis=1).add(1).cumprod()*100
        market=pd.DataFrame({'Close':m}).dropna()
    else: market=prices('KS11',start,end)
    if market.empty:return pd.DataFrame()
    p=st.progress(0,'종목 분석 준비 중...');s=st.empty();out=[];total=len(universe);done=0
    with ThreadPoolExecutor(max_workers=min(12,max(4,total//30 or 4))) as ex:
        fs=[ex.submit(analyze,row,start,end,market) for _,row in universe.iterrows()]
        for f in as_completed(fs):
            done+=1
            try:
                x=f.result()
                if x:out.append(x)
            except:pass
            p.progress(done/total if total else 1,text=f'종목 분석 중... {done:,}/{total:,} ({done/total*100:.1f}%)')
            s.info(f'분석 완료 {done:,}개 · 유효 데이터 {len(out):,}개')
    p.progress(1,'✅ 종목 검색 완료');s.success(f'검색 완료 · 성공 {len(out):,}/{total:,}')
    if not out:return pd.DataFrame()
    z=pd.DataFrame(out);z['상대순위점수']=.6*rank_pct(z.종합점수)+.4*rank_pct(z.내일진입점수);z['최종순위점수']=.55*z.종합점수+.30*z.내일진입점수+.15*z.상대순위점수;z['당일상대순위']=rank_pct(z.최종순위점수)
    return z.sort_values(['최종순위점수','종합점수'],ascending=False).reset_index(drop=True)


with st.sidebar:
    st.header('⚙️ 공통 전략 설정')
    min_score=st.slider('최종 관심 기준',50,90,70,help='전체 후보 중 우선 분석할 최소 점수입니다.')
    rank_limit=st.slider('허용할 종목 순위(%)',1,50,20,help='상위 몇 %까지 관심 종목으로 볼지 정합니다.')
    min_value=st.number_input('20일 평균 거래대금(원)',0,100_000_000_000,2_000_000_000,500_000_000)
    rsi_hot=st.slider('단기 과열 RSI 기준',65,85,72)
    ema_hot=st.slider('20일 평균 대비 최대 이격(%)',5,25,12)
    atr_hot=st.slider('가격 변동 위험 상한(%)',3,15,7)
    st.checkbox('하락 추세에서는 신규 매수 막기',True,key='block')
    st.checkbox('시장보다 강한 종목만 보기',True,key='strong')
    st.checkbox('상승 추세 종목만 보기',True,key='up')

st.subheader('🔎 오늘 어떤 종목을 살펴볼까요?')
market_choice=st.selectbox('분석할 시장',['🇰🇷 KOSPI + KOSDAQ 전체','KOSPI','KOSDAQ'])
count=st.number_input('분석할 종목 수 (0 = 전체)',0,3000,300,100)
patterns=st.multiselect('우선 살펴볼 상승 구조',['🚀 돌파 시작형','📈 상승 지속형','🔄 눌림 재상승형','📊 상승 추세형'],default=['🚀 돌파 시작형','📈 상승 지속형','🔄 눌림 재상승형','📊 상승 추세형'])
run=st.button('🚀 오늘의 종목 찾기',type='primary',use_container_width=True)

if run:
    L=listing()
    if L.empty:st.error('KRX 종목 목록을 불러오지 못했습니다.');st.stop()
    if market_choice=='KOSPI':u=L[L.Market.str.upper().str.contains('KOSPI',na=False)].copy()
    elif market_choice=='KOSDAQ':u=L[L.Market.str.upper().str.contains('KOSDAQ',na=False)].copy()
    else:u=L[L.Market.str.upper().str.contains('KOSPI|KOSDAQ',regex=True,na=False)].copy()
    if count>0:u=u.head(int(count))
    st.info(f'🔍 {market_choice} · 검색 대상 {len(u):,}개')
    end=dt.date.today();start=end-dt.timedelta(days=450)
    R=scan(u,start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'))
    if R.empty:st.error('분석 가능한 데이터가 없습니다.');st.stop()
    F=R[R.종합점수>=min_score].copy();F=F[F.당일상대순위>=100-rank_limit];F=F[F['평균거래대금20일']>=min_value]
    if st.session_state.block:F=F[F.시장국면!='하락 국면']
    if st.session_state.strong:F=F[F.시장대비20일>0]
    if st.session_state.up:F=F[F.추세점수>=60]
    if patterns:F=F[F.상승패턴.isin(patterns)]
    F=F[F.RSI<=rsi_hot];F=F[F.EMA20이격<=ema_hot];F=F[F.ATR비율<=atr_hot*1.5]
    st.session_state.R=R;st.session_state.F=F

if 'R' in st.session_state:
    R=st.session_state.R;F=st.session_state.F
    st.success(f'✅ 검색 완료 · 전체 {len(R):,}개 · 조건 통과 {len(F):,}개')
    st.header('🏆 오늘의 상위 후보')
    top=F.head(20)
    if top.empty:st.warning('현재 설정에서 조건을 만족하는 종목이 없습니다. 조건을 조금 완화해보세요.')
    else:
        cols=['종목명','종목코드','시장','판단','상승패턴','종합점수','종목자체점수','내일진입점수','당일상대순위','수익률20일','시장대비20일','RSI','ADX','거래량비율','EMA20이격','과열위험']
        st.dataframe(top[cols],use_container_width=True,hide_index=True)
        st.header('🟢 왜 이 종목인가?')
        name=st.selectbox('설명을 볼 종목',top.종목명.tolist())
        r=top[top.종목명==name].iloc[0]
        a,w=reasons(r)
        c1,c2,c3,c4=st.columns(4);c1.metric('종합점수',f'{r.종합점수:.1f}');c2.metric('종목 자체',f'{r.종목자체점수:.1f}');c3.metric('내일 진입',f'{r.내일진입점수:.1f}');c4.metric('판단',r.판단)
        st.markdown('### ✅ 추천하는 이유');[st.write('• '+x) for x in a] if a else st.write('뚜렷한 강점이 충분하지 않습니다.')
        st.markdown('### ⚠️ 관망/제외해야 할 이유');[st.write('• '+x) for x in w] if w else st.write('큰 단기 위험 신호가 없습니다.')
        st.markdown('### 🔬 과거 비슷한 상황')
        h1,h2,h3,h4,h5=st.columns(5);h1.metric('유사상황',f"{int(r.유사상황수):,}회");h2.metric('5일 평균',f"{r.평균5일:+.2f}%" if pd.notna(r.평균5일) else '-');h3.metric('10일 평균',f"{r.평균10일:+.2f}%" if pd.notna(r.평균10일) else '-');h4.metric('20일 평균',f"{r.평균20일:+.2f}%" if pd.notna(r.평균20일) else '-');h5.metric('20일 승률',f"{r['승률20일']:.1f}%" if pd.notna(r['승률20일']) else '-')
    st.header('🎯 상위 5% · 10% · 20% 기대값 비교')
    rows=[]
    for label,pct in [('상위 5%',.05),('상위 10%',.10),('상위 20%',.20),('전체',1)]:
        x=R.head(max(1,math.ceil(len(R)*pct)));rows.append({'선별구간':label,'종목수':len(x),'평균점수':x.종합점수.mean(),'평균진입점수':x.내일진입점수.mean(),'평균기대값20일':x.평균20일.mean(),'승률20일':x['승률20일'].mean()})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.header('📈 상승 구조별 성과')
    st.dataframe(R.groupby('상승패턴').agg(종목수=('종목명','count'),평균점수=('종합점수','mean'),평균기대값20일=('평균20일','mean'),평균승률20일=('승률20일','mean')).reset_index(),use_container_width=True,hide_index=True)
    st.header('🌏 시장 국면별 성과')
    st.dataframe(R.groupby('시장국면').agg(종목수=('종목명','count'),평균점수=('종합점수','mean'),평균기대값20일=('평균20일','mean'),평균승률20일=('승률20일','mean')).reset_index(),use_container_width=True,hide_index=True)
    st.header('📥 결과 저장')
    st.download_button('⬇️ 전체 결과 CSV',R.to_csv(index=False,encoding='utf-8-sig'),f'KRX_V9_8_{dt.date.today()}.csv','text/csv',use_container_width=True)
else:
    st.info('👆 시장을 선택하고 「오늘의 종목 찾기」를 눌러주세요.')
    st.markdown('''### V9.8 핵심
- KOSPI / KOSDAQ / 전체 동시검색
- 실제 검색 진행률 표시
- 종목 자체 점수와 내일 진입 점수 분리
- 상대강도 / 추세 / 모멘텀 / 거래량 / 돌파 / 눌림
- RSI / ADX / ATR / EMA 이격
- 과열·유동성·갭 위험 관리
- 상위 5% / 10% / 20% 자동 비교
- 상승 패턴 분류
- 「왜 이 종목인가?」 한국어 설명
- 추천 / 관망 / 제외 이유 표시
- 과거 유사상황 기대값 표시

**주의:** 이 프로그램은 투자 연구용이며 점수나 과거 성과가 미래 수익을 보장하지 않습니다.''')
