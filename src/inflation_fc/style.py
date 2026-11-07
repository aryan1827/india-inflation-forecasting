"""One colour per entity, used by both the static figures and the app, so a
model or variable keeps its colour on every chart."""

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#eef3fb"        # RBI tolerance band shading
NEUTRAL = "#8a8984"     # benchmarks and reference lines
NEUTRAL_2 = "#b5b4ae"

MODEL_COLORS = {
    "SARIMA": "#2a78d6",
    "VAR": "#eb6834",
    "Gradient boosting": "#1baf7a",
    "Random forest": "#6250d6",
    "Random walk": NEUTRAL,
    "Seasonal average": NEUTRAL_2,
    "Actual": TEXT,
}

VAR_COLORS = {
    "inflation": "#2a78d6",
    "repo": "#eb6834",
    "credit_yoy": "#1baf7a",
    "output_gap": "#eda100",
    "iip_yoy": "#eda100",
    "oil_yoy": "#e87ba4",
    "inr_yoy": "#6250d6",
}

LABELS = {
    "inflation": "CPI inflation",
    "repo": "Repo rate",
    "credit_yoy": "Credit growth",
    "iip_yoy": "IIP growth",
    "oil_yoy": "Brent (y-o-y)",
    "inr_yoy": "INR/USD (y-o-y)",
    "output_gap": "Output gap",
    "real_repo": "Real repo rate",
}
