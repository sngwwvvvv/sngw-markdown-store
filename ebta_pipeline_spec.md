# Evidence-Based Technical Analysis (EBTA) Trading Pipeline — Technical Specification (Codex-Friendly)

> 목적: David Aronson의 Evidence-Based Technical Analysis(EBTA) 철학을 **알고리즘/백테스트 파이프라인**으로 구현하기 위한 *구현 가능한 수준*의 기술 명세서이다.  
> 본 문서는 “설명”이 아니라 “구현 지시”이며, 모든 MUST 항목은 코드로 강제되어야 한다.

---

## 0. 용어/기호 정의 (Single Source of Truth)

### 0.1 시간 인덱스와 데이터
- 시계열 인덱스: `t = 0..T-1` (정렬된 시간)
- OHLCV: `O_t, H_t, L_t, C_t, V_t`
- 벤치마크(시장) 가격: `P_mkt,t` (필수)
- 로그수익률:
  \[
  r_{mkt,t} = \ln(P_{mkt,t}) - \ln(P_{mkt,t-1})
  \]
  \[
  r_{strat,t} = \text{(실현 수익률; 포지션·수수료 반영 후)}
  \]

### 0.2 신호 / 포지션 / 실행 지연
- 신호(의사결정):  
  \[
  Signal_t = f(S_t \mid \Theta) \in \{-1, 0, 1\}
  \]
- 포지션(보유 상태): `pos_t ∈ {-1,0,1}`  
- 실행 지연(Execution_Lag) 정의:
  - `Execution_Lag = 0`: `t`의 신호로 **t 종가(MOC)** 실행 (슬리피지/체결 리스크)
  - `Execution_Lag = 1`: `t`의 신호로 **t+1 시가** 실행 (갭 리스크)
- **룩어헤드 금지 규칙 (MUST)**  
  - `Signal_t`는 **t 시점까지**의 정보만 사용한다.
  - 실제 거래에 적용되는 포지션은 `pos_t = Signal_{t-Execution_Lag}` 형태로 **shift**되어야 한다.

---

## 1. 시스템 아키텍처 (Modules)

파이프라인은 아래 모듈을 순서대로 실행한다.

1) **Data Preprocessor**  
2) **Signal Engine**  
3) **Backtest Core (Vectorized)**  
4) **Walk-Forward Orchestrator**  
5) **Statistical Validator (WRC / SPA / MCP)**  
6) **Bias Correction & Shrinkage**  
7) **Reporting & Artifacts**

각 모듈은 **입력/출력 스키마**를 고정하며, 결과는 다음 모듈이 그대로 사용한다.

---

## 2. 데이터 스키마 및 입력 제약

### 2.1 입력 데이터 (MUST)
- `df` (pandas.DataFrame), DatetimeIndex(UTC 권장), columns:
  - `open, high, low, close, volume`
  - `benchmark_close` (시장 벤치마크 가격; 없으면 별도 입력으로 제공)
- 결측치 처리:
  - OHLCV 결측: 해당 bar 제거 또는 forward fill 정책을 명시(기본: 제거)
  - 벤치마크 결측: **전략 데이터와 동일한 인덱스로 정렬** 후 결측 제거(기본: outer join 후 dropna)

### 2.2 타임프레임 감지
- `bar_seconds` 또는 `pd.infer_freq` 기반으로 타임프레임을 결정한다.
- 코인(24/7)과 주식(세션) 구분이 필요하면, 세션 캘린더는 옵션으로 분리(본 명세서는 24/7을 기본 가정).

---

## 3. Detrending (Aronson Position-Bias Detrending)

### 3.1 목적
시장 장기 추세(베타)로 얻는 수익을 알파로 착각하지 않도록, **포지션 편향(Position Bias)**에 의해 발생 가능한 기대 수익을 제거한다.

### 3.2 핵심 수식 (MUST)
1) 시장 평균 로그수익률:
\[
\bar{r}_{mkt} = \frac{1}{T}\sum_{t=1}^{T} r_{mkt,t}
\]

2) 포지션 편향:
\[
PB = \frac{N_{long}-N_{short}}{N_{total}}
\]
- `N_total = T` (현금일 포함)
- `N_long = count(pos_t = 1)`, `N_short = count(pos_t = -1)`

3) 디트렌딩된 전략 수익률:
\[
r'_{strat,t} = r_{strat,t} - (pos_t \cdot \bar{r}_{mkt})
\]

> 구현상 `pos_t`는 **실제 보유 포지션(shift 반영 후)** 이어야 한다. (Signal이 아님)

### 3.3 함수 인터페이스 (MUST)
```python
def calculate_detrended_returns(
    strat_returns: "pd.Series",      # r_strat,t (fees/slippage 포함 후)
    mkt_returns: "pd.Series",        # r_mkt,t (log returns 권장)
    positions: "pd.Series"           # pos_t in {-1,0,1}, aligned index
) -> "pd.Series":
    """
    Returns:
        detrended_returns: pd.Series, r'_strat,t
    Constraints:
        - All series MUST share the same index.
        - positions MUST already reflect Execution_Lag (shifted).
    """
```

### 3.4 참고 (선택 보고)
- `expected_bias_return = PB * mean(mkt_returns)` 는 리포트용 보조 지표로 저장한다.

---

## 4. Signal Engine (Objective Rules)

### 4.1 신호 정의 (MUST)
- `Signal_t`는 결정론적 함수이며 랜덤 요소 금지.
- 신호 출력은 반드시 `{-1,0,1}`.
- 인과성(룩어헤드) 위반 금지:
  - 지표 계산 시 `rolling(window)`는 현재 시점까지 사용 가능하나, **실행 포지션은 shift로 지연**.

### 4.2 최소 구현 요구사항
Signal Engine은 다음을 반환해야 한다.
- `signals`: `pd.Series` in `{-1,0,1}`
- `features`: (옵션) 지표/필터 결과 DataFrame
- `rule_params`: 사용 파라미터 dict (Θ)

### 4.3 Execution Lag 적용 (MUST)
```python
pos = signals.shift(Execution_Lag).fillna(0).clip(-1, 1).astype(int)
```

---

## 5. Backtest Core (Vectorized)

### 5.1 단일 자산, 단일 포지션(±1) 기본
- 기본 가정:
  - 포지션은 `-1,0,1` (레버리지/포지션 사이징은 확장 포인트)
  - 체결가격 모델:
    - `Execution_Lag=0`: `fill_price_t = close_t`
    - `Execution_Lag=1`: `fill_price_t = open_t`
- 거래비용:
  - `fee_bps` (왕복/편도 정의 MUST 명시; 기본: 편도)
  - `slippage_bps` (편도)

### 5.2 수익률 계산 규칙 (MUST)
- 단순화된 bar-to-bar PnL:
  - `pos_t`를 보유한 상태에서 다음 bar 수익률을 얻는다.
- 예시(로그수익률 기반):
  \[
  r_{raw,t} = pos_{t-1}\cdot(\ln(C_t)-\ln(C_{t-1}))
  \]
  - 실행지연/체결 모델에 따라 `C` 대신 `fill_price`를 사용하도록 설계 가능.
- 비용 차감:
  - 거래 발생 시점: `trade_t = 1[pos_t != pos_{t-1}]`
  - 비용:
    \[
    cost_t = trade_t \cdot (fee\_bps + slippage\_bps)\cdot 10^{-4}
    \]
  - 최종:
    \[
    r_{strat,t} = r_{raw,t} - cost_t
    \]

### 5.3 출력 (MUST)
- `strat_returns` (pd.Series)
- `positions` (pd.Series)
- `trades` (pd.Series, 0/1)
- `equity_curve` (pd.Series; 누적 로그/단순 수익률 중 하나를 명시)

---

## 6. Walk-Forward (Rolling Window) Orchestrator

### 6.1 목적
비정상성(non-stationarity)을 반영하여 IS 최적화 → OOS 검증을 반복하고, 전체 기간 OOS 성과를 연결(concatenate)한다.

### 6.2 윈도우 정의
- Train(최적화): `[t-N, t]`
- Test(검증): `[t+1, t+k]`
- Step(전진): `step = k` 또는 별도 설정

### 6.3 권장 기본값 (Configuration)
> 아래 값은 “기간”이 아니라 “샘플 수 + 국면 커버”를 목표로 한다.  
> OOS에서 최소 거래 횟수(최소 샘플)를 만족하지 못하면 해당 fold를 무효 처리하거나 기간을 확장해야 한다.

| Timeframe | Train Window | Test Window | Step | Min Trades (OOS) |
|---|---:|---:|---:|---:|
| Daily | 3 years | 1 year | 6 months | ≥ 30 |
| 15m | 6 months (~4,000 bars) | 2 months (~1,300) | 1 month | ≥ 30 |
| 5m | 3 months (~6,000) | 1 month (~2,000) | 2 weeks | ≥ 50 |
| 1m | 4 weeks (~15,000) | 1 week (~3,900) | 3 days | ≥ 100 |

### 6.4 fold 출력 (MUST)
각 fold는 다음을 저장한다.
- 선택된 규칙/파라미터(최적화 결과)
- OOS 구간의 `signals, positions, strat_returns, detrended_returns`
- OOS 거래 횟수, 기본 성과 지표(평균/샤프/MaxDD 등)

---

## 7. Complexity Control (Overfitting Penalty)

### 7.1 복잡성 점수 정의 (MUST)
\[
C_{score} = \sum_i w_i \cdot N_i
\]

| Component | Weight (w) | Count (N) 예시 |
|---|---:|---|
| Continuous Params | 1.0 | MA 기간, RSI 임계값 |
| Logic Gates | 0.5 | AND/OR/IF |
| Filters | 1.5 | Volume/Volatility 필터 |
| Non-linear Transform | 2.0 | log/exp/square |

### 7.2 페널티 적용 성과지표 (MUST)
\[
Metric_{adj} = Metric_{raw}\cdot\left(1-\lambda\cdot\ln(C_{score})\right)
\]
- `λ` 기본값: `0.05` (프로젝트 설정으로 고정 가능)
- `C_score <= 1`일 때 로그 안정성 처리:
  - `ln(max(C_score, 1.0001))` 사용

---

## 8. Statistical Validator (핵심 엔진)

본 섹션은 “데이터 마이닝 편향(DMB)”을 제어하기 위한 **구현 필수** 알고리즘을 정의한다.

### 8.1 공통 입력: 성과 행렬 M (MUST)
- `M`: shape `(T, N)`
  - `T`: 시간(일/바)
  - `N`: 테스트한 규칙(또는 규칙×파라미터 조합) 개수
- `M[t, n] = r'_{n,t}`: n번째 규칙의 **디트렌딩된** 수익률 (MUST)

---

## 9. White’s Reality Check (WRC)

### 9.1 목적
수많은 규칙 중 최선의 성과가 우연인지 검정한다.  
귀무가설 \(H_0\): 모든 규칙의 기대 디트렌딩 수익률은 0.

### 9.2 관측 통계량
- 규칙별 평균 성과:
  \[
  \bar{f}_n = \frac{1}{T}\sum_{t=1}^{T} M[t,n]
  \]
- 관측된 최대 성과:
  \[
  \hat{f}_{max} = \max_n \bar{f}_n
  \]

### 9.3 부트스트랩 방식 (MUST)
- **Block Bootstrap** 사용 (자기상관/변동성 군집 보존)
- 블록 길이 `q` 기본:
  - Daily: `q ∈ [10, 20]`
  - Intraday: `q ≈ bars_per_day` (예: 1m crypto=1440)
- 구현 허용 옵션:
  - Circular Block Bootstrap (단순 구현)
  - Stationary Bootstrap (더 정교; 권장)

### 9.4 중심화(Centering) 단계 (MUST)
귀무가설을 강제하기 위해 부트스트랩 샘플을 중심화한다.

- `b`번째 부트스트랩 샘플에서:
  - resample index로 `M*` 생성
  - 규칙별 평균:
    \[
    \bar{f}^*_{n,b} = \text{Mean}(M^*[:,n]) - \bar{f}_n
    \]
  - 최대값:
    \[
    max^*_b = \max_n \bar{f}^*_{n,b}
    \]
- `D_null = {max^*_1,...,max^*_B}` 저장

### 9.5 p-value (MUST)
\[
p_{WRC} = \frac{\#\{b: max^*_b > \hat{f}_{max}\}}{B}
\]
- `B` 기본: 2000 (최소 1000)

### 9.6 WRC 함수 인터페이스 (MUST)
```python
def whites_reality_check(
    M: "np.ndarray",            # shape (T, N), detrended returns
    block_len: int,
    B: int = 2000,
    seed: int | None = 42
) -> dict:
    """
    Returns dict with:
        f_bar: (N,)
        f_max_hat: float
        null_dist: (B,)
        p_value: float
    """
```

---

## 10. Hansen’s SPA (Superior Predictive Ability)

### 10.1 목적
WRC의 보수성(매우 나쁜 모델이 분산을 키워 검정력 저하)을 개선한다.

### 10.2 스튜던트화(Studentization) (MUST)
- 규칙별 t-통계량 기반 비교:
  \[
  t_n = \frac{\bar{f}_n}{\hat{\sigma}_n/\sqrt{T}}
  \]
- \(\hat{\sigma}_n\)은 HAC(Newey-West) 또는 block-based 분산 추정 권장.

### 10.3 Thresholding (MUST)
성능이 현저히 낮은 규칙은 0으로 절단하여 노이즈를 줄인다.

- 임계값(명세 고정):
  \[
  \bar{f}_n < -\sqrt{\frac{\hat{var}_n}{T}}\cdot 2\ln\ln(T)\ \Rightarrow\ \bar{f}_n := 0
  \]

### 10.4 SPA 부트스트랩 (MUST)
- WRC와 동일한 block bootstrap을 사용하되,
  - 중심화 및 thresholding을 적용
  - 통계량은 `max t*_b` 형태로 구성
- p-value는 WRC와 동일한 비교 방식으로 산출

### 10.5 SPA 함수 인터페이스 (MUST)
```python
def hansen_spa_test(
    M: "np.ndarray",            # (T, N)
    block_len: int,
    B: int = 2000,
    seed: int | None = 42
) -> dict:
    """
    Returns dict with:
        t_stats: (N,)
        t_max_hat: float
        null_dist: (B,)
        p_value: float
    """
```

---

## 11. Monte Carlo Permutation (MCP)

### 11.1 목적
시장 수익률과 신호의 연결 고리를 끊어 “우연한 상관”을 평가한다.

### 11.2 알고리즘 (MUST)
- `signals/positions`는 고정
- 시장 수익률(또는 가격 변화)의 순서를 무작위로 섞는다.
  - 변동성 군집 보존을 위해 **Block Shuffle** 권장
- 섞인 시장 수익률에 고정된 포지션을 적용하여 가상 수익률을 계산
- 반복하여 성과 분포 생성

### 11.3 p-value (권장 정의)
- 관측된 성과(평균 디트렌딩 수익률 또는 샤프 등)가
- permutation 분포의 상위 \(\alpha\) (예: 5%)에 속하는지 평가:
  \[
  p_{MCP} = \frac{\#\{b: metric^*_b \ge metric_{obs}\}}{B}
  \]
- `B` 기본: 5000

---

## 12. Bias Correction & Shrinkage (Markowitz/Chu style)

### 12.1 데이터 마이닝 편향 추정 (MUST)
- WRC/SPA에서 생성된 `null_dist`는 “실력 0인 상태에서 최고의 규칙이 얻을 수 있는 성과” 분포.
- 편향 추정:
  \[
  Bias_{DM} \approx E[\text{null\_dist}]
  \]

### 12.2 보정 기대수익률 (MUST)
\[
E_{adjusted} = \hat{f}_{max} - \text{Mean}(\text{null\_dist})
\]
- SPA를 사용하는 경우 \(\hat{f}_{max}\) 대신 \(\hat{t}_{max}\)를 직접 보정하지 말고,
  - 최종적으로는 **수익률 단위**로 환산된 \(\hat{f}_{max}\)와 `null_dist`의 성과 단위를 일치시켜 적용한다.

### 12.3 채택 조건 (MUST)
- `E_adjusted - (transaction_cost + slippage)` 가 양수이며,
- WRC/SPA/MCP 중 최소 1개 이상에서 `p_value < 0.05` 를 만족할 때만 채택.

---

## 13. 리포팅/시각화 (MUST)

### 13.1 WRC/SPA Null Distribution 시각화
- 히스토그램 + 관측 통계량 위치(수직선)
- 저장 파일: `wrc_null_hist.png`, `spa_null_hist.png`

### 13.2 산출물(Artifacts)
- `oos_equity_curve.csv`
- `fold_summary.csv` (fold별 성과/거래수/파라미터)
- `validator_results.json` (p-value, null_dist summary, seed 등)

---

## 14. 재현성 / 난수 관리 (MUST)
- 모든 랜덤 프로세스는 `seed`를 받아 재현 가능해야 한다.
- numpy `Generator(PCG64)` 권장:
```python
rng = np.random.default_rng(seed)
```

---

## 15. 엣지 케이스 / 안전장치 (MUST)
- `T`가 너무 작아 `ln ln T`가 정의되지 않는 경우(SPA):
  - `T < 3`이면 테스트 중단 및 “insufficient samples” 반환
- 거래 횟수 부족:
  - OOS 최소 거래수 미달 fold는 제외하거나 윈도우 확장 (정책을 설정에 명시)
- `C_score`가 0/음수:
  - 최소값 클램프: `C_score = max(C_score, 1.0001)`
- 비용 모델:
  - fee/slippage는 편도/왕복 정의를 코드 상수로 고정하고 리포트에 표시

---

## 16. 구현 체크리스트 (Acceptance Criteria)

### 16.1 MUST
- [ ] `signals` 생성에서 룩어헤드 없음 (unit test: 미래 데이터 변경해도 과거 signal 불변)
- [ ] `positions = signals.shift(Execution_Lag)` 적용
- [ ] `strat_returns` 계산에 비용 반영
- [ ] `detrended_returns`가 3.2 수식과 일치
- [ ] `M (T,N)` 구성 및 WRC p-value 산출
- [ ] WRC에서 **Centering 단계** 구현 확인
- [ ] SPA에서 Studentization + Thresholding 구현
- [ ] MCP에서 permutation 분포 생성 및 p-value 산출
- [ ] Bias correction `E_adjusted` 산출 및 채택 조건 적용
- [ ] 리포트/아티팩트 생성

### 16.2 SHOULD
- [ ] 분산 추정에 HAC(Newey-West) 또는 block 기반 적용
- [ ] Stationary bootstrap 옵션 제공
- [ ] fold별 파라미터/복잡성 점수 로깅

---

## 17. 최소 코드 스켈레톤 (Optional, for Codex Bootstrapping)

> 아래는 “구현 시작점”이며, 최적화/규칙 생성 로직은 프로젝트에 맞게 추가한다.

```python
from __future__ import annotations
import numpy as np
import pandas as pd

def calculate_detrended_returns(strat_returns: pd.Series,
                                mkt_returns: pd.Series,
                                positions: pd.Series) -> pd.Series:
    idx = strat_returns.index.intersection(mkt_returns.index).intersection(positions.index)
    sr = strat_returns.loc[idx].astype(float)
    mr = mkt_returns.loc[idx].astype(float)
    pos = positions.loc[idx].astype(int).clip(-1, 1)

    mu_mkt = mr.mean()
    detr = sr - (pos * mu_mkt)
    return detr

def vectorized_backtest(df: pd.DataFrame,
                        signals: pd.Series,
                        execution_lag: int = 1,
                        fee_bps: float = 0.0,
                        slippage_bps: float = 0.0) -> dict:
    idx = df.index.intersection(signals.index)
    px = df.loc[idx, "close"].astype(float)
    pos = signals.loc[idx].shift(execution_lag).fillna(0).clip(-1, 1).astype(int)

    # log returns on close-to-close
    r = np.log(px).diff().fillna(0.0)
    raw = pos.shift(1).fillna(0).astype(int) * r

    trade = (pos != pos.shift(1)).astype(int).fillna(0)
    cost = trade * ((fee_bps + slippage_bps) * 1e-4)

    strat = raw - cost
    eq = strat.cumsum()  # log-equity
    return {"positions": pos, "trades": trade, "returns": strat, "equity": eq}

# WRC/SPA/MCP는 본 명세의 인터페이스에 따라 별도 구현
```

---

## 18. 확장 포인트 (Not required, but reserved)
- 포지션 사이징(연속 포지션), 레버리지, 리스크 패리티
- 다자산 포트폴리오(상관·공분산) + 리밸런싱
- 실행 엔진(주문 타입, 슬리피지 모델 고도화, 큐/체결 확률)

---

**End of Spec**
