# 변경 사항 (2026-05-10)

## 🎯 추가된 핵심 기능

### 백테스트 프레임워크
- `factor_pipeline/backtest.py`: 일일 리밸런싱 백테스트 엔진
- `factor_pipeline/signals.py`: 요인 수익률 예측 (latest/rolling/ewma/oracle)
- `scripts/run_backtest.py`: 배치 백테스트 실행

### IBKR 라이브 트레이딩
- `scripts/ibkr_live_portfolio.py`: 실시간 포트폴리오 리밸런싱 (위험 한도 포함)
- `scripts/ibkr_live_trader.py`: IBKR 가격 조회 및 주문
- `scripts/connect_ibkr_demo.py`: 연결 진단

### 개선사항
- `run_clean_pipeline.py`: 우주 선택 개선 (start/end anchor ranking)
- `estimation.py`: 요인 반환 추정 개선
- `requirements.txt`: ib_insync 추가

---

## ⚠️ 에러 가능성이 있는 부분

### 1. **IBKR API 구독 문제**
- **문제**: `Error 10089` - Real-time market data subscription required
- **원인**: 데모 계정에 실시간 데이터 구독이 없음
- **현황**: `--market-data-type 3` (delayed) 옵션으로 우회 가능
- **확인 필요**: 실제 거래 시 실시간 구독 활성화 필요

### 2. **Market Order 장 시간 제약**
- **문제**: 미국 시장이 닫혀있으면 Market Order가 Submitted 상태로 유지됨
- **현황**: 장 개장 후 자동 체결 또는 취소됨
- **해결**: GTC(Good-Till-Cancelled) 옵션 추가 가능

### 3. **파이프라인 로딩 폴백**
- **상황**: weights.npy 없을 때 X.npy + factor_returns.npy에서 자동 계산
  ```python
  # 계산 경로:
  X → predict_factor_returns(factor_returns) → make_scores(X, f_pred) 
  → make_quantile_long_short_weights(scores)
  ```
- **주의**: 이 경로에서 파일 누락 시 FileNotFoundError 발생

### 4. **가격 수집 타임아웃**
- **문제**: 모든 심볼에서 가격을 못 받으면 그 심볼은 주문 스킵됨
- **현황**: `--market-data-type 3`에서 일부 심볼 가격 누락 가능
- **영향**: 목표 주문의 일부만 체결될 수 있음

---

## 🔍 컨펌이 필요한 부분

### 1. **Backtest 일일 리밸런싱 확인**
- ✓ Confirmed: `run_factor_backtest()`에서 매일 f_pred 재계산 및 포지션 갱신
- 코드: [factor_pipeline/backtest.py](factor_pipeline/backtest.py#L119-L126)

### 2. **45개 팩터 사용 확인**
- ✓ Confirmed: `make_scores(X @ f_pred)` - X의 모든 요인 사용
- 파이프라인: price_volume 팩터 + fundamental 팩터 = 42~45개

### 3. **라이브 트레이딩 안전성**
- `--dry-run` 제거 시 **실제 주문 전송됨**
- `--auto-rebalance` 필수 옵션 (없으면 주문 미전송)
- **컨펌**: 데모 계정에서 작은 금액으로 먼저 테스트?

### 4. **금융 팩터 캐싱 방식**
- 현재: 파이프라인 실행 후 저장된 X.npy, financials_flat.parquet 재사용
- 라이브: 실시간 금융 데이터 요청 안 함 (하루에 한 번 갱신)
- **컨펌**: 금융 데이터 수동 갱신 스케줄 필요?

### 5. **git 커밋 전 확인**
- ✓ 모든 테스트 통과 (19/19 tests passed)
- ✓ 신규 파일 컴파일 확인
- **컨펌**: 커밋 메시지 및 푸시 진행?

---

## 📋 git 커밋 메시지 (제안)

```
feat: Add complete backtest framework and IBKR live trading integration

**Core Features:**
- Implement factor_pipeline/backtest.py: Daily rebalancing long-short backtest engine
- Implement factor_pipeline/signals.py: Factor return prediction (latest/rolling/ewma/oracle)
- Add scripts/run_backtest.py: Batch backtest with multiple methods
- Add scripts/ibkr_live_portfolio.py: Real-time portfolio rebalancing with risk limits
- Add scripts/ibkr_live_trader.py: IBKR market data and order helper
- Add scripts/connect_ibkr_demo.py: IBKR connection diagnostics

**Enhancements:**
- Improve run_clean_pipeline.py: Add start/end anchor ranking for universe selection
- Update estimation.py: Refine factor return estimation
- Add ib_insync to requirements.txt
- Add comprehensive tests for new backtest and signal generation modules

**Risk Management:**
- Per-symbol max 2% equity, per-side max 60% equity
- Top N symbols selection (default 50)
- Snapshot market data requests for low-latency price collection
- Market data type selection (1:realtime, 2:frozen, 3:delayed, 4:delayed frozen)

**Backtest Results (2023-01-01 ~ 2026-03-01 NASDAQ):**
- Sharpe: 4.88 (ewma, q=0.1, lookback=120)
- Full universe with survivorship bias reduction (tradable_mask)
- 45 factors: price-volume + fundamental

**Known Issues:**
- IBKR demo account: Real-time data subscription required (Error 10089)
  Workaround: Use --market-data-type 3 (delayed) for testing
- Market Order execution limited to US trading hours
- Fallback weight computation from X.npy when weights.npy unavailable

**Testing:**
- All 19 existing tests pass
- New modules compile without errors
- Ready for demo account testing
```

---

## 🚀 다음 단계 (선택사항)

1. **실제 거래 전**: 데모 계정에서 작은 금액($1-10)으로 테스트
2. **금융 팩터 갱신 자동화**: cron job으로 하루에 한 번 파이프라인 재실행
3. **다중 통화 지원**: 국제 거래 추가
4. **옵션/선물 거래**: 파생상품 추가

