"""
detrend_analysis.py
-------------------
Example template for a Python script:
- Organize imports
- Define functions
- Run analysis in a __main__ block
"""

# -----------------
# 1. Imports
# -----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# -----------------
# 2. Functions
# -----------------
def generate_dummy_data(n=30, seed=42):
    """Create synthetic time series data with a linear trend + noise."""
    np.random.seed(seed)
    years = np.arange(2000, 2000 + n)
    trend = 2.0 * years  # linear trend
    noise = np.random.normal(0, 100, size=n)
    values = trend + noise
    return pd.DataFrame({"year": years, "value": values})


def detrend_series(df, y_col="value", time_col="year"):
    """Fit a linear model and return residuals (detrended values)."""
    X = sm.add_constant(df[time_col])  # add intercept
    y = df[y_col]
    model = sm.OLS(y, X).fit()
    df[y_col + "_detrended"] = model.resid
    return df, model


def plot_detrended(df, y_col="value", time_col="year"):
    """Plot original vs. detrended series."""
    plt.figure(figsize=(10, 6))

    # Original
    plt.plot(df[time_col], df[y_col], label="Original", color="blue")

    # Detrended
    plt.plot(df[time_col], df[y_col + "_detrended"], label="Detrended", color="red")

    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title("Original vs. Detrended Series")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------
# 3. Main script
# -----------------
if __name__ == "__main__":
    # Step 1: Create data
    df = generate_dummy_data()

    # Step 2: Detrend
    df, model = detrend_series(df)

    # Step 3: Print summary
    print(model.summary())

    # Step 4: Plot results
    plot_detrended(df)
