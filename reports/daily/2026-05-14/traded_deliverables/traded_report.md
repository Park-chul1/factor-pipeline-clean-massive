# 2026-05-14 Traded Names Deliverables

Generated at 2026-05-15T02:11:31+00:00 from the refreshed 2026-05-14 close cache.

## Data Basis
- US trading date: 2026-05-14
- Final close bars: 12,121 tickers, 12,121 rows, volume sum 19,396,523,111
- Final after-close report run: 2026-05-15T02:10:13.517953+00:00
- Paper order snapshot submitted_at range: 2026-05-14T17:13:00.563471+00:00..2026-05-14T17:13:00.563471+00:00
- Residual heatmap window: 2026-04-01..2026-05-13; 2026-05-14 residual is unavailable because this workflow needs next-day return for residual computation.

## Order Status Summary
| order_status | orders | tickers | notional | estimated_total_cost |
| --- | --- | --- | --- | --- |
| Filled | 275 | 275 | $620,023.52 | $186.01 |
| Submitted | 198 | 198 | $421,393.34 | $126.42 |
| Inactive | 6 | 6 | $22,984.66 | $6.90 |
| PreSubmitted | 3 | 3 | $6,290.55 | $1.89 |
| Cancelled | 2 | 2 | $6,816.20 | $2.04 |

## Paper Side Summary
| side | orders | tickers | notional |
| --- | --- | --- | --- |
| BUY | 256 | 256 | $623,031.75 |
| SELL | 228 | 228 | $454,476.52 |

## Final Close Target Overlay
- Paper order rows: 484; unique traded tickers: 484
- Filled rows in snapshot: 275
- Gross paper notional in snapshot: $1,077,508.27
- Estimated total cost in snapshot: $323.25
- Final close non-zero target among traded tickers: 219/484
- Final report gross/net exposure: 0.9754 / 0.0110

| final_close_target_direction | orders | tickers | notional |
| --- | --- | --- | --- |
| FLAT | 265 | 265 | $501,590.26 |
| LONG | 122 | 122 | $361,166.89 |
| SHORT | 97 | 97 | $214,751.12 |

## Largest Paper Orders
| ticker | paper_side | quantity | limit_price | notional | order_status | final_close_target_direction | alpha_score | net_alpha_after_cost_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BWEN | BUY | 5647 | 3.9800 | 22,475.0600 | Filled | FLAT | -0.0022 | -25.0314 |
| MSFT | SELL | 50 | 407.6000 | 20,380.0000 | Filled | FLAT | 0.0002 | -0.5388 |
| PENG | BUY | 353 | 47.1900 | 16,658.0700 | Submitted | LONG | 0.0067 | 63.9276 |
| FLNC | BUY | 674 | 21.9300 | 14,780.8200 | Filled | LONG | 0.0056 | 52.9719 |
| LQDA | BUY | 123 | 58.8200 | 7,234.8600 | Submitted | LONG | 0.0077 | 73.9942 |
| MRAM | BUY | 174 | 39.9800 | 6,956.5200 | Filled | LONG | 0.0109 | 105.5163 |
| FDMT | BUY | 478 | 9.9700 | 4,765.6600 | Filled | LONG | 0.0087 | 83.9820 |
| DGXX | BUY | 549 | 8.6800 | 4,765.3200 | Cancelled | LONG | 0.0135 | 132.2322 |
| PTEN | BUY | 394 | 12.0900 | 4,763.4600 | Filled | LONG | 0.0055 | 52.4807 |
| WEST | BUY | 561 | 8.4900 | 4,762.8900 | Submitted | LONG | 0.0066 | 63.4215 |
| MU | BUY | 6 | 793.6900 | 4,762.1400 | Filled | LONG | 0.0104 | 100.6935 |
| NVTS | BUY | 209 | 22.7800 | 4,761.0200 | Filled | LONG | 0.0114 | 110.9551 |
| MOVE | SELL | 224 | 21.2400 | 4,757.7600 | Submitted | SHORT | 0.0041 | -44.0001 |
| CEVA | BUY | 125 | 38.0600 | 4,757.5000 | Filled | LONG | 0.0067 | 63.6846 |
| NUCL | BUY | 440 | 10.8100 | 4,756.4000 | Submitted | FLAT | 0.0045 | 41.9383 |
| ERII | SELL | 555 | 8.5700 | 4,756.3500 | Filled | SHORT | -0.0063 | 60.1459 |
| STFS | SELL | 459 | 10.3600 | 4,755.2400 | Inactive | SHORT | -0.0193 | 189.7050 |
| INOD | BUY | 52 | 91.4400 | 4,754.8800 | Submitted | LONG | 0.0096 | 92.8837 |
| AVAH | SELL | 646 | 7.3600 | 4,754.5600 | Filled | SHORT | -0.0055 | 52.0999 |
| SMX | SELL | 433 | 10.9800 | 4,754.3400 | Inactive | SHORT | -0.0407 | 403.9716 |

## Strongest Final-Close Long Alpha Among Traded Names
| ticker | paper_side | order_status | final_close_model_side | final_close_target_direction | target_weight | alpha_score | net_alpha_after_cost_bps | top_positive_factor_contributors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DGXX | BUY | Cancelled | SELL | LONG | 0.0045 | 0.0135 | 132.2322 | intraday_return;close_to_low_20;mom_21 |
| LWLG | BUY | Filled | BUY | LONG | 0.0008 | 0.0118 | 115.0016 | intraday_return;hl_range_20;mom_252 |
| NVTS | BUY | Filled | SELL | LONG | 0.0045 | 0.0114 | 110.9551 | mom_5;mom_252;mom_21 |
| AGPU | BUY | Submitted | SELL | LONG | 0.0045 | 0.0113 | 109.6502 | intraday_return;hl_range_20;volatility_20 |
| MRAM | BUY | Filled | SELL | LONG | 0.0045 | 0.0109 | 105.5163 | mom_5;close_to_low_20;mom_21 |
| AXTI | BUY | Filled | SELL | LONG | 0.0045 | 0.0107 | 103.5115 | mom_252;mom_63;hl_range_20 |
| MU | BUY | Filled | SELL | LONG | 0.0045 | 0.0104 | 100.6935 | mom_252;mom_21;mom_5 |
| GSIT | BUY | Filled | SELL | LONG | 0.0045 | 0.0102 | 98.9816 | intraday_return;mom_5;hl_range_20 |
| INOD | BUY | Submitted | SELL | LONG | 0.0045 | 0.0096 | 92.8837 | mom_5;close_to_low_20;mom_21 |
| LXRX | BUY | Filled |  | FLAT | 0.0000 | 0.0096 | 92.8622 | mom_5;intraday_return;mom_252 |

## Strongest Final-Close Short Alpha Among Traded Names
| ticker | paper_side | order_status | final_close_model_side | final_close_target_direction | target_weight | alpha_score | net_alpha_after_cost_bps | top_negative_factor_contributors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SMX | SELL | Inactive | COVER | SHORT | -0.0045 | -0.0407 | 403.9716 | parkinson_vol_20;intraday_return;mom_5 |
| EZGO | BUY | Submitted |  | FLAT | 0.0000 | -0.0285 | -287.7532 | parkinson_vol_20;mom_5;volatility_10 |
| ASBP | BUY | Submitted | SHORT | SHORT | -0.0008 | -0.0259 | 255.9820 | parkinson_vol_20;intraday_return;volatility_63 |
| ADTX | SELL | Filled |  | FLAT | 0.0000 | -0.0252 | -254.9054 | parkinson_vol_20;volume_z_60;mom_5 |
| PHOE | SELL | Inactive | COVER | SHORT | -0.0037 | -0.0233 | 229.8757 | parkinson_vol_20;volatility_10;mom_5 |
| BNZI | SELL | Submitted | COVER | SHORT | -0.0045 | -0.0199 | 196.3836 | parkinson_vol_20;intraday_return;volatility_10 |
| STFS | SELL | Inactive | COVER | SHORT | -0.0045 | -0.0193 | 189.7050 | parkinson_vol_20;mom_10;volatility_10 |
| AKAN | SELL | Submitted | COVER | SHORT | -0.0045 | -0.0184 | 180.9404 | parkinson_vol_20;intraday_return;mom_5 |
| HTCO | SELL | Inactive | COVER | SHORT | -0.0045 | -0.0181 | 177.8403 | parkinson_vol_20;overnight_return;volatility_10 |
| CUE | SELL | Filled | COVER | SHORT | -0.0045 | -0.0167 | 164.4638 | parkinson_vol_20;mom_10;volatility_10 |

## Cancelled / Inactive / PreSubmitted Snapshot Rows
| ticker | paper_side | quantity | limit_price | notional | order_status | final_close_target_direction | alpha_score | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DGXX | BUY | 549 | 8.6800 | 4,765.3200 | Cancelled | LONG | 0.0135 |  |
| WYFI | BUY | 68 | 30.1600 | 2,050.8800 | Cancelled | SHORT | -0.0036 |  |
| HTCO | SELL | 637 | 7.4600 | 4,752.0200 | Inactive | SHORT | -0.0181 |  |
| KXIN | SELL | 693 | 5.7400 | 3,977.8200 | Inactive | SHORT | 0.0015 |  |
| PHOE | SELL | 282 | 14.0800 | 3,970.5600 | Inactive | SHORT | -0.0233 |  |
| PSIG | SELL | 107 | 7.2400 | 774.6800 | Inactive | FLAT | -0.0015 |  |
| SMX | SELL | 433 | 10.9800 | 4,754.3400 | Inactive | SHORT | -0.0407 |  |
| STFS | SELL | 459 | 10.3600 | 4,755.2400 | Inactive | SHORT | -0.0193 |  |
| ERNA | SELL | 465 | 10.2100 | 4,747.6500 | PreSubmitted | SHORT | -0.0040 |  |
| IPST | SELL | 101 | 7.6500 | 772.6500 | PreSubmitted | SHORT | -0.0139 |  |
| ZVRA | SELL | 65 | 11.8500 | 770.2500 | PreSubmitted | FLAT | 0.0002 |  |

## Files
- traded_report.md: this report
- traded_residual_heatmap.png: top 80 traded tickers by absolute residual z-score over the latest 30 residual dates
- traded_orders_with_final_close_model.csv: paper order rows merged with final close model fields
- traded_residuals.csv: all residual rows for all traded tickers
- nonfilled_or_working_orders.csv: all rows not marked Filled in the paper order snapshot
- traded_status_summary.csv, traded_side_summary.csv, final_close_target_direction_summary.csv: compact summaries
