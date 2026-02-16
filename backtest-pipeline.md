# 증거 기반 기술적 분석(EBTA) 파이프라인의 고도화 및 알고리즘 구현을 위한 기술 명세서

---

## 1. 서론 
승산 없는 게임에서의 생존 전략금융 시장은 근본적으로 불확실성과 노이즈로 가득 찬 복잡계입니다. 20년 이상 프롭 트레이딩 데스크(Proprietary Trading Desk)에서 퀀트 전략을 운용하며 목격한 가장 흔한 실패는 시장의 신호를 잘못 해석하는 것이 아니라, 무작위적인 노이즈를 패턴으로 착각하는 '데이터 마이닝 편향(Data Mining Bias)'에서 비롯됩니다. 데이비드 아론슨(David Aronson) 박사의 Evidence-Based Technical Analysis (EBTA)는 이러한 편향을 통계적으로 제어하고, 기술적 분석을 주관적인 예술의 영역에서 객관적인 과학의 영역으로 끌어올리는 중요한 방법론을 제시합니다. 

귀하가 제출한 마크다운 파일은 아론슨의 핵심 철학인 '객관적 규칙(Objective Rules)', '데이터 분할(Data Splitting)', '복잡성 제어(Complexity Control)', '통계적 검증(Statistical Verification)'의 4대 기둥을 잘 포착하고 있습니다. 그러나 실제 알고리즘 트레이딩 시스템, 특히 AI 어시스턴트가 환각(Hallucination) 없이 코드를 생성할 수 있는 수준의 '기술 명세(Technical Specification)'로서는 구체적인 수식과 로직이 부족합니다. 단순히 "검증한다"는 지시만으로는 데이터 스누핑(Data Snooping)을 방지할 수 있는 정교한 부트스트랩(Bootstrap) 로직을 구현할 수 없습니다.

본 보고서는 귀하의 초안을 바탕으로, 실제 SMB Capital 수준의 운용이 가능한 백테스트 파이프라인을 구축하기 위해 필요한 수학적, 통계적 디테일을 보완합니다. 특히 **포지션 편향(Position Bias)에 기반한 디트렌딩(Detrending) 수식, White's Reality Check(WRC)와 Hansen's SPA(Superior Predictive Ability)의 구체적 알고리즘, 그리고 마코위츠/추(Markowitz/Chu)의 수익률 축소(Shrinkage) 공식**을 명확히 정의합니다. 이를 통해 단순히 과거 데이터를 설명하는 모델이 아니라, 미래의 불확실성 속에서도 견고한 예측력을 가지는 '알파(Alpha)'를 발굴하는 시스템을 설계하고자 합니다.

---

## 2. 방법론적 기반과 데이터 엔지니어링의 정밀화

### 2.1 객관적 규칙과 이진 신호의 제약 조건
EBTA의 출발점은 해석의 여지를 0으로 만드는 것입니다. 전통적인 기술적 분석이 "추세가 강해 보인다"와 같은 주관적 판단에 의존한다면, EBTA는 이를 엄격한 논리 회로로 변환합니다.

#### 2.1.1 신호의 결정론적 정의 (Deterministic Definition)
트레이딩 규칙은 시장 데이터 상태 벡터 $S_t$를 입력받아 트라이너리(Trinary) 신호 ${-1, 0, 1}$를 출력하는 결정론적 함수 $f$여야 합니다.

$$Signal_t = f(O_t, H_t, L_t, C_t, V_t | \Theta) \in \{-1, 0, 1\}$$


여기서 $\Theta$는 파라미터 집합(예: 이동평균 기간)입니다.

AI 어시스턴트가 이를 코딩할 때 가장 중요한 것은 **인과성(Causality)의 위배 금지**입니다. $t$ 시점의 종가($C_t$)를 사용하여 생성된 신호는 반드시 $t+1$ 시점의 시가($O_{t+1}$) 또는 $t$ 시점의 장마감 동시호가(MOC)에만 실행 가능해야 합니다.
- 보완 사항: 제출된 파일은 '매수/매도/중립'으로만 정의되어 있으나, 실제 구현 시에는 **Execution Lag(실행 지연)** 파라미터를 명시해야 합니다.
  - Execution_Lag = 0: MOC 주문 (슬리피지 리스크 존재
  - Execution_Lag = 1: 다음 날 시가 주문 (갭 리스크 존재)

### 2.2 디트렌딩(Detrending): 알파와 베타의 분리
아론슨 방법론의 핵심이자 귀하의 초안에서 가장 보완이 필요한 부분은 바로 데이터의 디트렌딩입니다. 단순히 평균을 0으로 만드는 것(Zero-centering)만으로는 충분하지 않습니다. 아론슨은 전략이 시장의 장기적인 상승 추세(Beta)에 편승하여 수익을 낸 것을 실력(Alpha)으로 착각하지 않도록, **포지션 편향(Position Bias)** 을 제거한 수익률을 검증에 사용해야 한다고 강조합니다.

#### 2.2.1 시장 수익률과 전략 수익률의 분해
전략의 총 수익률 $R_{total}$은 다음과 같이 분해될 수 있습니다.
$$R_{total} = R_{alpha} + R_{beta} + R_{noise}$$
우리가 검증하고자 하는 것은 오직 $R_{alpha}$입니다. 
시장이 연 10% 상승하는 동안 단순히 매수 포지션만 유지한 전략이 10% 수익을 냈다면, 이는 예측력($R_{alpha}$)이 0인 전략입니다.

#### 2.2.2 포지션 편향 기반 디트렌딩 공식 (Mathematical Specification)
AI가 구현해야 할 정확한 디트렌딩 로직은 다음과 같습니다.
1. **시장 벤치마크의 일별 평균 수익률 ($\bar{r}_m$) 계산**:
전체 백테스트 기간 $T$ 동안의 시장 로그 수익률 평균을 구합니다.
$$\bar{r}_m = \frac{1}{T} \sum_{t=1}^{T} r_{m,t}$$

2. **전략의 포지션 편향 ($PB$) 계산**:
전략이 시장에 노출된 방향성의 평균을 구합니다.
$$PB = \frac{N_{long} - N_{short}}{N_{total}}$$
- $N_{long}$: 매수 포지션 보유 일수
- $N_{short}$: 매도 포지션 보유 일수
- $N_{total}$: 전체 거래 일수 (현금 보유일 포함)

3. **기대 편향 수익률 ($E_{bias}$) 산출**:
전략이 아무런 예측력 없이 단순히 동전 던지기로 포지션을 잡았을 때, 시장 추세에 의해 얻게 될 기대 수익입니다.
$$E_{bias} = PB \times \bar{r}_m$$

4. **디트렌딩된 전략 수익률 ($r'_{s,t}$) 도출**:
매 시점 $t$의 전략 수익률에서 시장의 평균적 추세 성분을 제거합니다.
$$r'_{s,t} = r_{s,t} - (Signal_{t-1} \times \bar{r}_m)$$

또는 전체 성과에서 차감할 경우:
$$R'_{total} = R_{total} - (T \times E_{bias})$$

**핵심 인사이트**: 이 과정을 거치면, 상승장에서 단순히 매수만 들고 있는 전략의 초과 수익은 0으로 수렴하게 됩니다. 즉, 이 디트렌딩된 수익률이 양수(+)여야만 해당 전략이 시장의 변동성(Volatility)을 이용하여 진정한 알파를 창출했다고 볼 수 있습니다. 
AI 코딩 시 이 공식을 함수화(calculate_detrended_returns)하여 백테스트 엔진의 핵심 모듈로 탑재해야 합니다.

---

## 3. 워크 포워드 분석과 복잡성 제어

### 3.1 롤링 윈도우(Rolling Window) 워크 포워드
데이터 분할에 있어 귀하의 보고서는 Training/Testing/Validation의 3단 구성을 언급했습니다. 이를 더욱 강화하여 **Anchored vs. Rolling** 방식을 구분해야 합니다. 금융 시계열은 비정상성(Non-stationarity)을 가지므로, 너무 오래된 데이터는 현재의 시장 미시구조(Market Microstructure)를 반영하지 못할 수 있습니다.
- 권장 구현: 롤링 윈도우 방식
  - Train (최적화): $t-N$ ~ $t$ (예: 3년)
  - Test (검증 및 복잡성 선택): $t+1$ ~ $t+k$ (예: 1년)
  - 이 윈도우를 $k$ 기간만큼 전진시키며 전체 기간의 OOS(Out-of-Sample) 수익 곡선을 연결(Concatenate)합니다.

#### 3.1.1 타임프레임별 권장 윈도우 설정 (Configuration Table)
일중(Intraday) 거래, 특히 코인이나 선물과 같은 고빈도 데이터에서는 '기간'보다 '샘플 수(거래 횟수)'와 '시장 국면(Regime)' 커버리지가 중요합니다. OOS 기간 내에 최소 30~50회 이상의 거래가 발생해야 통계적 검증이 가능합니다.

|타임프레임 (Timeframe)|In-Sample (Train, 최적화)|Out-of-Sample (Test, 검증)|Step (업데이트 주기)|비고 (Rationale)|
|----------|-----|------|-------|--------------|
|일봉 (Daily)|3년|1년|6개월|전통적인 스윙 트레이딩 기준|
|15분 (15 min)|6개월 (약 4,000 bars)|2개월 (약 1,300 bars)|1개월|실적 시즌 등 계절성을 포함하고 충분한 거래 샘플 확보|
|5분 (5 min)|3개월 (약 6,000 bars)|1개월 (약 2,000 bars)|2주|변동성 군집(Volatility Clustering) 주기(2~3주) 커버|
|1분 (1 min)|4주 (약 15,000 bars)|1주 (약 3,900 bars)|3일|빠른 미시구조 변화 반영 및 요일 효과 검증 (최소 1주 OOS)|

- 주의: 1분 봉의 경우 OOS를 1주 미만(예: 3일)으로 잡을 경우 요일 효과(Weekend Effect 등)에 의해 데이터가 편향될 수 있으므로 최소 1주일을 권장합니다.

### 3.2 복잡성 페널티(Complexity Penalty)와 오버피팅
아론슨은 모델의 자유도(Degrees of Freedom)가 높을수록 데이터 마이닝 편향이 기하급수적으로 증가한다고 경고합니다. 따라서 '최적의 복잡성'을 찾는 과정은 단순히 Test 셋의 수익률이 꺾이는 지점을 찾는 것을 넘어, **오캄의 면도날(Occam's Razor)** 원칙을 수식으로 적용해야 합니다.

#### 3.2.1 복잡성 점수 ($C_{score}$) 산출 로직
AI가 전략을 평가할 때 다음의 기준으로 복잡성 점수를 매겨야 합니다.

|요소 (Component)|가중치 (Weight)|예시|
|---------------|--------------|----|
|연속형 파라미터 (Continuous Params)|1.0|이동평균 기간, RSI 임계값|
|논리 연산자 (Logic Gates)|0.5|AND, OR, IF-THEN|
|필터 조건 (Filters)|1.5|거래량 필터, 변동성 필터|
|비선형 변환 (Non-linear Transform)|2.0|제곱, 로그, 지수 변환|

$$C_{score} = \sum w_i \times N_i$$

#### 3.2.2 성과 조정 (Penalized Performance Metric)
단순 수익률이 아닌, 복잡성으로 할인된 성과지표를 최적화의 기준으로 삼아야 합니다. 정보 기준(Information Criterion)과 유사한 접근입니다.

$$Metric_{adj} = Metric_{raw} \times (1 - \lambda \times \ln(C_{score}))$$

여기서 $\lambda$는 과적합에 대한 민감도 계수입니다. 이 수식을 적용하면, 복잡한 규칙은 압도적으로 높은 수익을 내지 않는 한 간단한 규칙보다 낮은 점수를 받게 되어 자연스럽게 과적합을 방지할 수 있습니다.

---

## 4. 데이터 마이닝 편향의 통계적 교정 (핵심 엔진)
이 부분이 보고서의 심장부입니다. 단순히 "통계적 검증을 한다"는 문장만으로는 불충분합니다. 귀하의 요청대로 `White's Reality Check(WRC)`, `Hansen's SPA`, `Monte Carlo Permutation(MCP)`의 구체적인 구현 로직을 상세히 기술합니다. 이는 AI가 `scipy`나 `numpy`를 활용해 직접 구현할 수 있는 수준이어야 합니다.

### 4.1 데이터 마이닝 편향(DMB)의 정의
수천 개의 규칙을 테스트하여 가장 좋은 것을 고르는 행위는 필연적으로 '운(Luck)'을 '실력(Skill)'으로 가장하게 만듭니다. 관측된 최고의 성과($f_{best}$)는 진정한 성과($\mu_{best}$)와 편향($Bias$)의 합입니다.

$$f_{best} = \mu_{best} + Bias_{mining}$$

우리의 목표는 $Bias_{mining}$을 추정하여 제거하거나, 귀무가설($H_0$: 최고의 규칙도 예측력이 없다)을 기각하는 것입니다.

### 4.2 White's Reality Check (WRC) 알고리즘 명세
White(2000)가 제안한 이 방법은 부트스트랩을 통해 귀무가설 하에서의 최대 성과 분포를 생성합니다.

**AI 구현을 위한 의사코드(Pseudocode) 로직:**
1. **성과 행렬 구성 ($M$):**
    - $N$개의 규칙(Rule)에 대해 $T$ 기간 동안의 일별 디트렌딩 수익률을 계산하여 $T \times N$ 행렬을 생성합니다.
    - 실제 관측된 각 규칙의 평균 수익률 벡터 $\bar{f}$를 계산합니다.
    - 가장 높은 성과를 낸 규칙의 수익률 $\hat{f}_{max} = \max(\bar{f})$를 저장합니다.

2. **부트스트랩 루프 (반복 횟수 $B=1000$ 이상):**

    - **Circular Block Bootstrap**: 금융 시계열의 자기상관(Autocorrelation)을 보존하기 위해 단순 복원 추출 대신 블록 단위로 데이터를 추출합니다.

    - **Block Length ($q$) 설정**:
        - 일봉 데이터: 통상 10~20일.
        - 인트라데이 (1분/5분) 데이터: 자기상관성이 길게 유지되므로 하루(1 Day) 단위로 설정해야 합니다. 
        (예: 24시간 코인 시장의 1분 봉 = 1440 bars). 이는 일중 변동성 패턴(Volatility Pattern)을 보존하기 위함입니다.

    - $b$번째 부트스트랩 샘플에 대해 성과 행렬 $M^*$를 생성합니다.
    - **중심화(Centering - 중요!)**: 귀무가설($E[f]=0$)을 강제하기 위해 부트스트랩 평균에서 원본 평균을 뺍니다.
    $$\bar{f}^*_{n, b} = \text{Mean}(M^*_{n, b}) - \bar{f}_n$$
    - 이 중심화된 수익률 중 **최댓값**을 찾습니다.
    $$\text{max}^*_b = \max_{n=1..N} (\bar{f}^*_{n, b})$$
    - 이 $\text{max}^*_b$를 분포 리스트 $D_{null}$에 저장합니다.
    
3. **P-Value 산출:**
    - $D_{null}$ 분포에서 실제 관측된 $\hat{f}_{max}$보다 큰 값이 나올 확률을 계산합니다.
    $$\text{p-value}_{WRC} = \frac{\text{Count}(\text{max}^*_b > \hat{f}_{max})}{B}$$
    
    **해석:** P-value가 0.05 미만이라면, 우리가 발견한 최고의 규칙은 95%의 신뢰수준에서 "운이 좋은 수만 개의 규칙 중 하나"가 아니라 "진짜 예측력을 가진 규칙"이라고 판단할 수 있습니다.

### 4.3 Hansen's SPA (Superior Predictive Ability) 알고리즘 명세
Hansen(2005)의 SPA는 WRC의 단점(성능이 매우 나쁜 모델이 분산에 영향을 주어 검증력을 떨어뜨리는 문제)을 개선한 것입니다.

**AI 구현을 위한 차이점:**
1. **스튜던트화(Studentization)**: 수익률 평균을 표준편차로 나누어 t-통계량 기반으로 비교합니다.
2. **임계값(Thresholding) 적용**: 부트스트랩 과정에서 성능이 현저히 낮은 모델($\bar{f}_n < -\sqrt{\frac{\hat{var}_n}{T}} \times 2 \ln \ln T$)은 0으로 간주하여 노이즈를 제거합니다.
3. 이로 인해 SPA는 WRC보다 검증력(Power)이 높으며, 덜 보수적인 결과를 제공합니다.

### 4.4 Monte Carlo Permutation (MCP) 알고리즘 명세
MCP는 부트스트랩과 달리 **시장 데이터와 신호 간의 연결 고리를 끊는 방식**입니다.

**AI 구현 로직:**
1. 규칙의 매매 신호 벡터 $S$는 고정합니다.
2. 시장 수익률 벡터 $R_{mkt}$의 순서를 무작위로 섞습니다(Shuffle). (단, 변동성 군집 현상을 반영하려면 역시 Block Shuffle을 사용해야 함).
3. 섞인 시장 데이터에 고정된 신호를 적용하여 가상의 수익률을 계산합니다.
4. 이 과정을 5000회 반복하여 "무작위 시장에서의 성과 분포"를 만듭니다.
5. 실제 성과가 이 분포의 상위 5% 안에 드는지 확인합니다.

---

## 5. 수익률 축소(Shrinkage)와 최종 기대 수익률 산출
테스트를 통과했다고 해서 백테스트 수익률($R_{backtest}$)을 그대로 기대 수익률로 믿어서는 안 됩니다. "승자의 저주(Winner's Curse)"로 인해, 선택된 최적의 모델은 필연적으로 상향 편향되어 있습니다.

### 5.1 Markowitz/Chu의 편향 보정 공식
아론슨은 Markowitz와 Xu(1994), 그리고 관련 연구들을 인용하여 관측된 수익률에서 데이터 마이닝 편향을 차감할 것을 제안합니다.
$$E_{real} = R_{best} - Bias_{DM}$$

여기서 $Bias_{DM}$은 WRC/SPA 과정에서 생성된 $D_{null}$ 분포의 평균값으로 추정할 수 있습니다. 즉, **"아무런 실력이 없는 수만 개의 모델 중에서 우연히 1등이 낼 수 있는 평균적인 수익률"** 을 실제 수익률에서 빼주는 것입니다.

$$Bias_{DM} \approx E[\text{max}^*_b]$$

이 과정을 거친 $E_{real}$이 여전히 거래비용을 상회하는 양수일 때만 실제 트레이딩에 투입합니다.

---

## 6. 알고리즘 구현을 위한 기술 명세서
다음은 AI 어시스턴트에게 입력하여 즉시 코딩을 시작할 수 있도록, 위에서 논의된 모든 수학적, 논리적 엄밀함을 포함하여 재작성된 마크다운 파이프라인 명세서입니다.

### **Evidence-Based Trading System Pipeline**
본 문서는 David Aronson의 Evidence-Based Technical Analysis 방법론을 기반으로 한 퀀트 트레이딩 백테스트 엔진의 기술 명세서(Technical Specification)이다. 개발자(또는 AI)는 아래의 수식과 로직을 엄수하여 구현해야 한다.

#### 1. System Architecture Overview
파이프라인은 다음의 순차적 모듈로 구성된다.
1. **Data Preprocessor**: 데이터 정제 및 Aronson식 디트렌딩
2. **Signal Engine**: 객관적 규칙 기반의 이진 신호 생성
3. **Backtest Core**: 매트릭스 기반의 고속 연산 (Vectorized Backtest)
4. **Statistical Validator**: WRC/SPA 및 MCP를 통한 유의성 검증
5. **Performance Adjuster**: 데이터 마이닝 편향을 제거한 기대 수익률 산출

#### 2. Module Specifications

**Module 1: Data Preprocessor & Detrending**

**목표**: 시장의 베타(Beta)를 제거하고 순수한 알파(Alpha)만을 분리하기 위한 시계열 변환.
- **Log Returns Calculation**:$$r_t = \ln(P_t) - \ln(P_{t-1})$$

- **Aronson's Position Bias Detrending (필수 구현)**:단순 평균 차감이 아닌, 전략의 포지션 편향을 고려한 디트렌딩을 수행한다.
```Python

def calculate_detrended_returns(strategy_returns, market_returns, positions):
    """
    strategy_returns: pd.Series (전략의 일별 수익률)
    market_returns: pd.Series (벤치마크의 일별 수익률)
    positions: pd.Series {-1, 0, 1} (전략의 포지션 상태)
    """
    # 1. 전체 기간 시장 평균 수익률
    mu_market = market_returns.mean()

    # 2. 포지션 편향 (Long 비율 - Short 비율)
    # 0인 구간(현금)은 분모(Total Days)에는 포함되나 분자에는 영향 없음
    n_total = len(positions)
    n_long = (positions > 0).sum()
    n_short = (positions < 0).sum()
    position_bias = (n_long - n_short) / n_total

    # 3. 편향에 의한 기대 수익률
    expected_bias_return = position_bias * mu_market

    # 4. 디트렌딩 (각 시점의 수익률에서 편향 성분을 제거)
    # 주의: Aronson은 개별 시점 보정보다 전체 성과 보정을 선호하나, 
    # 시계열 분석을 위해 매일의 Beta 성분을 제거하는 방식을 권장함.
    detrended_returns = strategy_returns - (positions * mu_market)

    return detrended_returns
```

**Module 2: Signal Engine & Complexity Control**

**목표**: 룩어헤드 편향(Look-ahead Bias)이 없는 결정론적 신호 생성 및 복잡성 제한.
- **Configuration Constants (Timeframe Dependent):** AI는 입력 데이터의 타임프레임을 감지하여 아래의 파라미터를 적용해야 한다.

|Timeframe|Train Window|Test Window|Minimum Trades (OOS)|
|---------|------------|-----------|--------------------|
|15 min|6 Months|2 Months|> 30|
|5 min|3 Months|1 Month|> 50|
|1 min|4 Weeks|1 Week|> 100|

- **Signal Lagging**: $t$ 시점의 종가($C_t$)를 이용해 계산된 지표는 반드시 $t$ 시점의 주문(MOC) 혹은 $t+1$ 시점의 주문에만 사용되어야 한다.
    - 코드 구현 시 `signal = raw_signal.shift(1)`을 강제하여 $t$일의 데이터로 $t+1$일의 수익을 낸다는 것을 명시할 것.

- **Universe Generation (Data Mining Space)**:
최적의 파라미터를 찾기 위해 탐색하는 모든 경우의 수를 기록해야 한다.
    - 예: MA(5, 10,... 200) $\times$ RSI(14, 21) $\times$ Breakout(True, False)
    - 이 $N$개의 모든 변형 규칙(Variant Rules)은 폐기되지 않고 'Validator' 모듈로 전달되어야 한다. (승자의 저주 계산용)

**Module 3: Statistical Validator (White's Reality Check)**

**목표:** 데이터 마이닝 편향을 보정한 P-value 산출. 부트스트랩 블록 길이는 데이터 특성에 맞게 조정된다.

- **Algorithm (Bootstrap Reality Check):**
1. **Input**: $T \times N$ 매트릭스 (T: 시간, N: 전체 테스트된 규칙들의 디트렌딩 수익률).
2. **Performance Vector**: 각 규칙의 평균 수익률 $\bar{f}$ 계산.
3. **Best Rule**: $\hat{f}_{max} = \max(\bar{f})$.
4. **Bootstrap Loop ($B=2000$)**:
    - `Stationary Bootstrap` (Politis & Romano, 1994) 사용. (평균 블록 길이 $q$ 설정, 예: $q=10$일).
    - Block Length ($q$) Determination:
        - IF `Timeframe` == 'Daily': $q \approx 10$ to $20$.
        - IF `Timeframe` < '1h' (Intraday): $q \approx \text{Bars per Day}$ (e.g., 1440 for 1m Crypto, 390 for 1m Stocks).
        - `Rationale`: 일중 계절성과 변동성 군집을 보존하기 위함.
    - 재샘플링된 인덱스로 수익률 매트릭스 $M^*$ 생성.
    - `Centering (중심화)`: $M^*_{centered} = M^* - \bar{f}$ (각 규칙의 부트스트랩 평균에서 원본 평균을 뺌).
    - 각 부트스트랩 샘플 내에서 최대 수익률 $\text{max}^*_b$을 찾음.

5. **P-value**:
$$p = \frac{\sum_{b=1}^{B} I(\text{max}^*_b > \hat{f}_{max})}{B}$$
($I$는 지시 함수)

**Module 4: Bias Correction & Shrinkage**

**목표**: 과대평가된 기대 수익률 보정.

- **Markowitz/Chu Shrinkage**:
    - 관측된 최대 수익률 $\hat{f}_{max}$에서 부트스트랩 분포의 평균(Data Mining Bias)을 차감.
    - $$E_{adjusted} = \hat{f}_{max} - \text{Mean}(\{\text{max}^*_1, \dots, \text{max}^*_B\})$$
    - 이 $E_{adjusted}$가 거래 비용(Transaction Cost)과 슬리피지(Slippage)를 제하고도 양수일 때만 전략을 채택.
    
#### 3. Implementation Constraints for AI

1. **Vectorization**: Python의 `pandas`와 `numpy`를 사용하여 루프를 최소화할 것.
2. **Reproducibility**: 모든 난수 생성(Random Seed)은 고정 가능해야 함.
3. **Visualization**: WRC 결과로서 'Null Distribution'의 히스토그램과 실제 $\hat{f}_{max}$의 위치를 시각화하는 코드를 포함할 것.

---

## 7. 결론 및 제언
본 보고서와 기술 명세서는 귀하가 기존에 작성한 마크다운 파일의 개념적 틀을 실제 금융 시장에서 작동 가능한 공학적 설계도로 변환한 것입니다. 특히 **포지션 편향 디트렌딩과 White's Reality Check의 중심화(Centering) 단계**는 대다수의 아마추어 퀀트들이 놓치는 부분이며, 이것이 바로 '백테스트에서는 부자, 실전에서는 파산'을 가르는 결정적인 차이입니다.

우리는 시장을 이길 확률을 구하는 것이 아니라, 우리가 틀렸을 확률(P-value)을 엄격하게 제어함으로써 생존합니다. 이 명세서를 통해 귀하의 AI 어시스턴트는 단순한 코더가 아니라, 엄격한 리스크 관리자가 되어 견고한 트레이딩 시스템을 구축할 수 있을 것입니다.

이 파이프라인의 다음 단계로는, 검증된 전략들 간의 **상관관계(Correlation)를 고려한 포트폴리오 최적화**와 **실시간 주문 집행(Order Execution) 알고리즘의 통합**을 권장합니다.