# R replication of the core Python results
# -----------------------------------------
# Re-runs the stationarity tests, the SARIMA model and the VAR on the same
# processed panel, and writes a side-by-side comparison with Python.
#
#   install.packages(c("forecast", "tseries", "urca", "vars", "lmtest", "sandwich", "jsonlite"))
#   Rscript r/replication.R          (run from the project root)

suppressPackageStartupMessages({
  library(forecast)
  library(tseries)
  library(urca)
  library(lmtest)
  library(sandwich)
  library(jsonlite)
})

panel <- read.csv("data/processed/india_macro_monthly.csv", stringsAsFactors = FALSE)
panel$date <- as.Date(panel$date)
out_dir <- "r/output"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

to_ts <- function(x, dates) {
  ok <- !is.na(x)
  first <- dates[ok][1]
  ts(x[ok], start = c(as.integer(format(first, "%Y")), as.integer(format(first, "%m"))), frequency = 12)
}

# ---- 1. Stationarity ---------------------------------------------------------
vars_to_test <- intersect(c("log_cpi", "inflation", "repo", "output_gap", "iip_yoy",
                            "oil_yoy", "inr_yoy", "credit_yoy"), names(panel))
stat_rows <- lapply(vars_to_test, function(v) {
  x <- na.omit(panel[[v]])
  adf <- suppressWarnings(adf.test(x))            # tseries: ADF with trend
  kp  <- suppressWarnings(kpss.test(x, null = "Level"))
  ers <- ur.df(x, type = "drift", selectlags = "AIC")  # urca: ADF with drift, AIC lags
  data.frame(variable = v,
             adf_trend_p = adf$p.value,
             adf_drift_stat = ers@teststat[1],
             adf_drift_cv5 = ers@cval[1, "5pct"],
             kpss_p = kp$p.value)
})
stat_r <- do.call(rbind, stat_rows)
write.csv(stat_r, file.path(out_dir, "stationarity_r.csv"), row.names = FALSE)
cat("\nStationarity (R):\n"); print(stat_r, digits = 3)

# ---- 2. Phillips curve with Newey-West errors ---------------------------------
lag1 <- function(x) c(NA, head(x, -1))
pc_data <- data.frame(
  inflation = panel$inflation,
  inflation_lag1 = lag1(panel$inflation),
  output_gap_lag1 = lag1(panel$output_gap),
  oil_yoy_lag1 = lag1(panel$oil_yoy),
  inr_yoy_lag1 = lag1(panel$inr_yoy),
  covid = panel$covid
)
pc <- lm(inflation ~ ., data = na.omit(pc_data))
pc_nw <- coeftest(pc, vcov = NeweyWest(pc, lag = 12, prewhite = FALSE, adjust = FALSE))
cat("\nPhillips curve, Newey-West SEs (R):\n"); print(pc_nw, digits = 3)
write.csv(as.data.frame(unclass(pc_nw)), file.path(out_dir, "phillips_r.csv"))

# ---- 3. SARIMA: same order as Python, plus auto.arima -------------------------
cfg <- fromJSON("results/model_config.json")
y <- to_ts(panel$log_cpi, panel$date)
# Python estimates on 100 * log CPI, so do the same here to compare coefficients.
fit_same <- Arima(100 * y, order = cfg$sarima_order, seasonal = cfg$sarima_seasonal_order[1:3],
                  method = "ML")
py_par <- read.csv("results/sarima_params.csv")
par_cmp <- data.frame(r = c(coef(fit_same), sigma2 = fit_same$sigma2))
par_cmp$python <- py_par$estimate
cat("\nSARIMA coefficients, R vs Python (last row: innovation variance):\n"); print(par_cmp, digits = 4)
write.csv(par_cmp, file.path(out_dir, "sarima_params_python_vs_r.csv"))
fit_auto <- auto.arima(y, d = 1, D = 1, stepwise = FALSE, approximation = FALSE)
cat("\nSARIMA with Python's order:\n"); print(fit_same)
cat("\nauto.arima choice:\n"); print(fit_auto)
checkresiduals(fit_same, plot = FALSE)

# 12-month forecast of y-o-y inflation from the log CPI path
fc <- forecast(fit_same, h = 12, level = 80)
hist_log <- as.numeric(y)
path <- c(hist_log, as.numeric(fc$mean) / 100)
n <- length(hist_log)
infl_fc <- 100 * (exp(path[(n + 1):(n + 12)] - path[(n + 1 - 12):(n + 12 - 12)]) - 1)

py_fc <- read.csv("results/sarima_future_ci.csv")
cmp <- data.frame(date = py_fc$date, python = py_fc$forecast, r = infl_fc,
                  diff = infl_fc - py_fc$forecast)
cat("\nSARIMA forecast, Python vs R (y-o-y %):\n"); print(cmp, digits = 3)
cat("Small differences come from different optimisers; large ones mean the likelihood is flat",
    "(e.g. an AR and MA term nearly cancelling) and the model is weakly identified.\n")
write.csv(cmp, file.path(out_dir, "sarima_python_vs_r.csv"), row.names = FALSE)

# ---- 4. VAR (needs the 'vars' package) ----------------------------------------
if (requireNamespace("vars", quietly = TRUE)) {
  library(vars)
  vinfo <- fromJSON("results/var_info.json")
  vdat <- na.omit(panel[, c(vinfo$variables, "covid")])
  exo <- as.matrix(vdat[, "covid", drop = FALSE])
  endo <- vdat[, vinfo$variables]
  sel <- VARselect(endo, lag.max = 6, type = "const", exogen = exo)
  cat("\nVAR lag selection (R):\n"); print(sel$selection)
  v <- VAR(endo, p = vinfo$lags, type = "const", exogen = exo)
  cat("\nVAR stability: max root modulus =", max(roots(v)), "\n")
  ir <- irf(v, impulse = "repo", n.ahead = 24, ortho = TRUE, boot = TRUE, runs = 500, ci = 0.90)
  shock_size <- ir$irf$repo[1, "repo"]
  r_irf <- data.frame(h = 0:24, inflation = ir$irf$repo[, "inflation"] / shock_size)
  py_irf <- read.csv("results/var_irf_repo.csv")
  py_irf <- py_irf[py_irf$response == "inflation", c("h", "irf")]
  irf_cmp <- merge(r_irf, py_irf, by = "h")
  names(irf_cmp) <- c("h", "r_inflation_response", "python_inflation_response")
  cat("\nInflation response to a 100 bp repo shock, R vs Python:\n"); print(head(irf_cmp, 13), digits = 3)
  write.csv(irf_cmp, file.path(out_dir, "var_irf_python_vs_r.csv"), row.names = FALSE)
  for (cause in vinfo$variables) {
    g <- causality(v, cause = cause)$Granger
    cat(sprintf("Granger: %-11s -> others  p = %.3f\n", cause, g$p.value))
  }
} else {
  cat("\nPackage 'vars' not installed; skipping the VAR. install.packages('vars') to run it.\n")
}

cat("\nDone. Outputs in", out_dir, "\n")
