# Results summary

Sample: 2014-01 to 2025-12 (monthly). Built 2026-11-07.

## Stationarity

- `log_cpi`: Unit root (ADF p=0.758, KPSS p=0.010, Zivot-Andrews break 2021-09)
- `d_log_cpi`: Stationary (ADF p=0.000, KPSS p=0.100, Zivot-Andrews break 2019-08)
- `inflation`: Inconclusive (ADF p=0.575, KPSS p=0.100, Zivot-Andrews break 2019-08)
- `d_inflation`: Stationary (ADF p=0.000, KPSS p=0.100, Zivot-Andrews break 2023-07)
- `repo`: Unit root (ADF p=0.463, KPSS p=0.010, Zivot-Andrews break 2022-04)
- `d_repo`: Stationary (ADF p=0.000, KPSS p=0.100, Zivot-Andrews break 2020-05)
- `output_gap`: Stationary (ADF p=0.000, KPSS p=0.100, Zivot-Andrews break 2020-02)
- `iip_yoy`: Stationary (ADF p=0.000, KPSS p=0.100, Zivot-Andrews break 2021-02)
- `oil_yoy`: Inconclusive (ADF p=0.071, KPSS p=0.100, Zivot-Andrews break 2022-11)
- `d_oil_yoy`: Stationary (ADF p=0.001, KPSS p=0.100, Zivot-Andrews break 2022-03)
- `inr_yoy`: Inconclusive (ADF p=0.098, KPSS p=0.075, Zivot-Andrews break 2014-02)
- `d_inr_yoy`: Stationary (ADF p=0.003, KPSS p=0.100, Zivot-Andrews break 2018-01)

## Phillips curve (OLS, Newey-West errors)

- R² = 0.812, n = 143
- Inflation persistence (lag coefficient) = 0.890
- Output gap effect: 0.008 (p = 0.102); long run 0.077
- Oil pass-through: -0.0001 (p = 0.962); long run -0.001 pp per 1% oil
- Exchange-rate pass-through: long run -0.008 pp per 1% depreciation
- Chow test at 2016-09: F = 1.36, p = 0.237
- Breusch-Godfrey p = 0.000, Breusch-Pagan p = 0.725, max VIF = 1.5

## Taylor rule

- Smoothing ρ = 0.982
- Long-run response to inflation gap = 1.55 (satisfies the Taylor principle, >1)
- Long-run response to output gap = 0.27
- Chow test at 2016-09: p = 0.759

## Models

- SARIMA(0, 1, 1)x(0, 1, 1, 12) on log CPI, chosen by BIC among models with white-noise residuals, on pre-2018 data
- VAR(1) on Brent (y-o-y), Output gap, CPI inflation, Repo rate, INR/USD (y-o-y); stable = True

## Monetary transmission (VAR, 100 bp repo shock)

- Brent (y-o-y): peak response -13.30 pp after 11 months (outside the 90% band around zero)
- Output gap: peak response -0.32 pp after 10 months (inside the 90% band around zero)
- CPI inflation: peak response -0.31 pp after 11 months (inside the 90% band around zero)
- INR/USD (y-o-y): peak response -0.98 pp after 18 months (inside the 90% band around zero)

## Forecast accuracy (rolling origin from 2018-01)

| h | model | RMSE | MAE | RMSE vs random walk |
|---|---|---|---|---|
| 1 | SARIMA | 0.58 | 0.41 | 0.73 |
| 1 | Seasonal average | 0.60 | 0.45 | 0.76 |
| 1 | Random forest | 0.68 | 0.49 | 0.87 |
| 1 | Gradient boosting | 0.76 | 0.56 | 0.97 |
| 1 | Random walk | 0.78 | 0.59 | 1.00 |
| 1 | VAR | 0.83 | 0.61 | 1.06 |
| 3 | SARIMA | 1.07 | 0.86 | 0.80 |
| 3 | Seasonal average | 1.11 | 0.87 | 0.83 |
| 3 | Gradient boosting | 1.22 | 0.99 | 0.92 |
| 3 | Random forest | 1.23 | 0.99 | 0.92 |
| 3 | Random walk | 1.33 | 1.07 | 1.00 |
| 3 | VAR | 1.50 | 1.19 | 1.13 |
| 12 | SARIMA | 1.97 | 1.63 | 0.82 |
| 12 | Seasonal average | 2.22 | 1.82 | 0.92 |
| 12 | Gradient boosting | 2.22 | 1.73 | 0.92 |
| 12 | Random forest | 2.24 | 1.79 | 0.93 |
| 12 | VAR | 2.32 | 1.87 | 0.97 |
| 12 | Random walk | 2.41 | 1.96 | 1.00 |

Best model per horizon (all months / excluding COVID):

- h=1: SARIMA (RMSE 0.58, 27% below random walk, DM p = 0.001) / SARIMA
- h=3: SARIMA (RMSE 1.07, 20% below random walk, DM p = 0.007) / SARIMA
- h=12: SARIMA (RMSE 1.97, 18% below random walk, DM p = 0.107) / SARIMA

## Outlook

- SARIMA forecast for 2026-12: 5.1% (80% interval 2.2 to 8.1)
