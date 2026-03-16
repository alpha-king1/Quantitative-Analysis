# Quantitative Market Analysis Framework

A modular **Python research framework** for systematic analysis of financial market data.

This repository demonstrates how quantitative research pipelines can be structured to analyze historical price data, generate market structure events, and evaluate hypotheses using statistical methods.

> ⚠️ This repository intentionally excludes proprietary trading strategies, models, or datasets.

---

## Project Objectives

The goal of this framework is to support structured quantitative research by enabling:

- Historical market data processing
- Market structure event generation
- Feature engineering on price series
- Statistical hypothesis testing
- Forward outcome analysis
- Distribution and regime analysis

---

## System Architecture

```
Quantitative Analysis Framework
│
├── data_pipeline
│   ├── data ingestion
│   ├── cleaning & normalization
│   └── resampling
│
├── feature_engineering
│   ├── rolling statistics
│   ├── volatility metrics
│   └── derived features
│
├── event_engine
│   ├── structural events
│   ├── liquidity events
│   └── signal extraction
│
├── research_engine
│   ├── hypothesis testing
│   ├── forward window analysis
│   └── distribution evaluation
│
└── analysis
    ├── baseline distribution comparison
    ├── regime behaviour analysis
    └── signal performance evaluation
```

---

## Research Workflow

```
Load Historical Market Data
        ↓
Clean and Normalize Dataset
        ↓
Generate Market Structure Events
        ↓
Extract Meaningfull Research Features
        ↓
Run Statistical Hypothesis Tests
        ↓
Evaluate Forward Outcome Distributions
        ↓
Analyze Statistical Significance
```

---

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Jupyter Notebook**

---

## Example Research Applications

This framework can support analyses such as:

- Event-based price reaction studies
- Forward return distribution analysis
- Market regime identification
- Statistical validation of trading signals

---

## Repository Purpose

The purpose of this project is to demonstrate the **engineering and research infrastructure used for quantitative financial analysis**.

The focus is on:

- Research pipeline design
- Structured experimentation
- Data analysis workflows
- Statistical evaluation methods

---

## Future Improvements

Planned improvements include:

- Modular backtesting engine
- Experiment tracking system
- Visualization tools for research analysis
- Expanded feature engineering library

---

## Author

**John Oduyebo**

Backend Engineer • Quantitative Data Analyst

GitHub: https://github.com/alpha-king1  
LinkedIn: https://linkedin.com/in/john-oduyebo-514923213
