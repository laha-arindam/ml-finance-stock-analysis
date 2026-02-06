# Machine Learning for Financial Stock Analysis

This repository presents an applied machine learning analysis of historical stock price data for **Apple Inc. (AAPL)**. The project combines traditional technical analysis with supervised machine learning models to study short-term stock return behavior.

The work is implemented as a single end-to-end Jupyter notebook and was developed as a final project for an MBA-level Machine Learning for Finance course.


## Project Objective

The goal of this project is to evaluate whether basic machine learning classifiers can improve upon a traditional technical trading rule (Bollinger Bands) when predicting short-term stock price movements.

The prediction task is framed as a **binary classification problem**:
- **1** → next-day return is positive  
- **0** → next-day return is non-positive  


## Dataset

- **Asset:** Apple Inc. (AAPL)
- **Source:** Yahoo Finance
- **Frequency:** Daily historical data
- **Variables include:** Open, High, Low, Close prices

The dataset is processed to compute returns and commonly used technical indicators.


## Feature Engineering

The following features are constructed from price data:

- Daily log returns
- 20-day Simple Moving Average (SMA)
- Rolling volatility (20-day standard deviation)
- Bollinger Bands (±2 standard deviations)
- Relative Strength Index (RSI, 14-day)

These indicators serve as inputs to the machine learning models.


## Baseline Strategy

A **Bollinger Bands trading rule** is used as a baseline strategy.  
This provides a traditional technical analysis benchmark against which machine learning models are compared.


## Machine Learning Models

Two supervised classification models are implemented:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression (L2 regularization)**

Key modeling details:
- Features are standardized using `StandardScaler`
- Hyperparameters are selected using `GridSearchCV`
- Models are trained using a chronological train/test split to avoid look-ahead bias


## Evaluation Methodology

### Classification Performance
Models are evaluated using:
- Accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

### Trading Strategy Evaluation
Model predictions are converted into simple trading signals:
- Long position if predicted return is positive
- Cash otherwise  
Signals are shifted forward by one day to prevent data leakage.

Performance metrics include:
- Total cumulative return
- Mean daily return
- Daily return volatility
- Sharpe-like ratio (mean / standard deviation)


## Key Results (Test Period)

| Strategy | Total Return | Sharpe-like Ratio |
|--------|-------------|------------------|
| Buy & Hold | ~34.7% | ~0.14 |
| Bollinger Bands | ~-8.4% | ~-0.14 |
| KNN Strategy | ~6.8% | ~0.05 |
| Logistic Regression Strategy | ~21.1% | ~0.14 |

**Observation:**  
Logistic Regression improves upon the Bollinger Bands strategy and KNN in this sample period, though it does not outperform buy-and-hold in terms of total return.


## Repository Contents

```text
ml-finance-stock-analysis/
├── data/
│   └── AAPL Yahoo.csv          # Historical AAPL price data (Yahoo Finance)
├── code/
│   └── Arindam_Project.ipynb   # Main analysis notebook
└── README.md                   # Project documentation
```



## How to Use

- Open `code/Arindam_Project.ipynb` in Jupyter Notebook or Google Colab
- Run all cells sequentially to reproduce the analysis and results
- The notebook contains all data preprocessing, modeling, and evaluation steps


## Limitations

- Daily stock direction prediction is inherently noisy
- Results are sample-period dependent
- Cross-validation uses standard K-Fold rather than time-series-specific splits
- No transaction costs or slippage are modeled


## Disclaimer

This project is for **educational and demonstration purposes only** and does not constitute financial or investment advice.
