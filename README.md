# Research on Stock Selection Strategies Based on Wind API and Local Backtesting Framework

## Project Overview

This project was developed during my internship at COFCO Futures as a research-oriented quantitative strategy project.

Under the guidance of my mentor and based on his practical industry experience, I conducted research on stock selection strategies built around a set of technical indicators. The strategy framework mainly includes three dimensions of indicators:

- technical indicators for stock selection
- technical indicators for sell timing
- technical indicators for buy timing

The project reflects a relatively risk-averse preference. Therefore, during the stock screening stage, the strategy tends to prioritize large-cap stocks that also satisfy certain turnover rate and trading volume constraints.

Some of the buy and sell signal logic can be found in the code. However, due to company confidentiality requirements, the version uploaded here is a simplified and partially redacted version, and does not include all technical indicators used in the original research.

---

## Strategy Features

- Research-oriented stock selection strategy based on technical indicators
- Separate indicator design for stock selection, buy timing, and sell timing
- Risk-averse stock universe filtering logic
- Preference for large-cap stocks with liquidity constraints
- Local backtesting framework for evaluating strategy performance

---

## Methodology

The strategy research in this project is mainly organized into three parts:

### 1. Stock Selection Indicators
These indicators are used to define the candidate stock pool.  
The strategy tends to focus on stocks with:

- relatively large market capitalization
- sufficient turnover rate
- adequate trading volume
- other technical conditions defined in the model

### 2. Buy Timing Indicators
These indicators are used to determine when a stock in the candidate pool becomes a valid entry opportunity.

### 3. Sell Timing Indicators
These indicators are used to determine when to exit a position after entry.

Together, these three layers form the core structure of the strategy.

---

## Notes on the Code

To comply with company confidentiality requirements, the code provided in this repository is a redacted version of the original project.

- Some technical indicators have been omitted
- Certain implementation details have been simplified
- The uploaded code is intended only to demonstrate the general research framework and workflow

---

## Requirements

Before using this project, please make sure the following conditions are met:

- Wind Financial Terminal is installed locally
- Wind Terminal is running properly
- your account has permission to access the WindPy API

Otherwise, the program may fail when attempting to retrieve financial data.

---

## Usage

After confirming that Wind Terminal and WindPy API permissions are available, run the main script locally:

```bash
python 选股模型.py
