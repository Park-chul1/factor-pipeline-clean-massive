# Daily Factor Report - 2026-05-12

## Model Decision
Generated close-based signals for execution on the next trading day. Proposed orders: 551. Expected turnover: 0.1000.

## Buy Candidates
| ticker | side | target_weight | alpha_score | reason_text |
| --- | --- | --- | --- | --- |
| NVTS | BUY | 0.00046511627906976747 | 0.009895157182002555 | BUY because predicted alpha is in the top decile. Main contributors were hl_range_20, mom_21, mom_252. Liquidity filter passed. |
| FLNC | BUY | 0.00046511627906976747 | 0.008003684373034182 | BUY because predicted alpha is in the top decile. Main contributors were mom_5, hl_range_20, mom_21. Liquidity filter passed. |
| INOD | COVER | -0.001930366984809786 | 0.007652298668922083 | COVER because predicted alpha is in the bottom decile. Main contributors were volatility_10, mom_10, parkinson_vol_20. Liquidity filter passed. |
| ABSI | BUY | 0.00046511627906976747 | 0.006089176843196468 | BUY because predicted alpha is in the top decile. Main contributors were hl_range_20, mom_21, close_to_low_20. Liquidity filter passed. |
| VICR | BUY | 0.00046511627906976747 | 0.005925242439893294 | BUY because predicted alpha is in the top decile. Main contributors were mom_252, hl_range_20, mom_21. Liquidity filter passed. |
| RILY | BUY | 0.00046511627906976747 | 0.005668730982102934 | BUY because predicted alpha is in the top decile. Main contributors were sales_to_price, mom_252, debt_to_equity. Liquidity filter passed. |
| LASR | BUY | 0.00046511627906976747 | 0.005609837191302935 | BUY because predicted alpha is in the top decile. Main contributors were mom_252, mom_5, hl_range_20. Liquidity filter passed. |
| TRAW | BUY | 0.00046511627906976747 | 0.005546272862619215 | BUY because predicted alpha is in the top decile. Main contributors were hl_range_20, intraday_return, close_to_high_20. Liquidity filter passed. |
| MBX | BUY | 0.00046511627906976747 | 0.0055200362786532375 | BUY because predicted alpha is in the top decile. Main contributors were mom_5, intraday_return, hl_range_20. Liquidity filter passed. |
| RFIL | BUY | 0.00046511627906976747 | 0.005494981551658824 | BUY because predicted alpha is in the top decile. Main contributors were mom_252, mom_5, mom_21. Liquidity filter passed. |

## Sell / Short Candidates
| ticker | side | target_weight | alpha_score | reason_text |
| --- | --- | --- | --- | --- |
| TDIC | SHORT | -0.00046511627906976747 | -0.02771423966591005 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volume_z_60, volatility_10. Liquidity filter passed. |
| UBXG | SHORT | -0.00046511627906976747 | -0.02723044644798721 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volume_z_60, volatility_63. Liquidity filter passed. |
| AREB | SHORT | -0.00046511627906976747 | -0.023958933606402888 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volatility_63, mom_5. Liquidity filter passed. |
| FCHL | SHORT | -0.00046511627906976747 | -0.023236153517255997 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volatility_63, mom_5. Liquidity filter passed. |
| CREG | SHORT | -0.00046511627906976747 | -0.022729664545651486 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volatility_63, volatility_10. Liquidity filter passed. |
| STFS | SHORT | -0.00046511627906976747 | -0.0162586878595836 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volatility_10, mom_10. Liquidity filter passed. |
| AUUD | SHORT | -0.00046511627906976747 | -0.015089929358461365 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volatility_63, mom_21. Liquidity filter passed. |
| BWEN | SHORT | -0.00046511627906976747 | -0.014799630849453705 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volume_z_60, volatility_10. Liquidity filter passed. |
| STAK | SHORT | -0.00046511627906976747 | -0.014344254427958949 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volatility_63, volatility_10. Liquidity filter passed. |
| OSRH | SHORT | -0.00046511627906976747 | -0.014109130720058122 | SHORT because predicted alpha is in the bottom decile. Main contributors were parkinson_vol_20, volatility_10, intraday_return. Liquidity filter passed. |

## Quantile Summary
| bucket | number_of_stocks | average_alpha | long_selected_count | short_selected_count |
| --- | --- | --- | --- | --- |
| 1 | 758 | -0.00471384228687899 | 5 | 222 |
| 2 | 4878 | -4.4773150394417846e-05 | 0 | 16 |
| 3 | 428 | 0.00035927546664668765 | 7 | 11 |
| 4 | 758 | 0.0013773157252866327 | 34 | 11 |
| 5 | 758 | 0.003421795068055666 | 229 | 16 |

## Risk / Exposure
Gross: 0.9907; Net: 0.0000; Long names: 275; Short names: 276.

## Data Quality
Warnings: Historical universe is from saved processed files; verify point-in-time membership before relying on historical comparisons.. Missing and finite factor coverage are recorded in pipeline_metadata.json.

## Timing Convention
Signals use data available at or before signal_date close. Orders are for the next trading day open. No intraday alpha recomputation is part of this workflow.
