# Health Analytics and Decision-Tree
This repository contains the analysis of profitability for home health agencies in the U.S., focusing on Medicare reimbursement and operational efficiency using a Decision Tree model

# Home Health Services Profitability Analysis

## Overview

This project explores the profitability of home health agencies in the U.S., focusing on Medicare reimbursement and operational strategies. The goal is to provide actionable insights for a Stillwater-based for-profit agency to guide its strategic expansion.

## Data Exploration

We analyzed a dataset with 45 variables and over 11,000 observations using Python's pandas library and ydata_profiling for Exploratory Data Analysis (EDA). Key findings include:

- **Profitability Classification**: Agencies were classified as profitable (1) or non-profitable (0) based on whether their profit was above or below the average.
- **Geographical Insights**: Texas, Florida, California, Illinois, and Ohio had the highest claim submissions.
- **Regional Analysis**: Region 6 showed the highest number of claims.

## Decision Tree Model

A decision tree classifier was used to predict profitability. Key aspects include:

- **Model Performance**:
  - Accuracy: 65%
  - Better at predicting class 1 (profitability) with higher precision, recall, and F1-score.
- **Confusion Matrix**:
  ```plaintext
  [[ 675  702]
   [ 416 1520]]

## Decile Analysis

- Top 70% of data contains 60% of class 1 values.
- Focus on the last 30% for improving profitability.

## Feature Selection

Top features influencing profitability include:
- **Provider_ID**: Significant impact with a score of 0.028936.

## Rules for Profitability

To predict class 1 (Profitability), the following conditions must be met:
- AvgHHAVPENonLUPA ≤ 1.75
- AvgSTVPENonLUPA ≤ 0.05
- AvgSNVPENonLUPA ≤ 17.05
- Provider_ID ≤ 152892.50
- AtrialFib ≤ 0.16
- HispBen ≤ 24.50
- TotLUPAEpis ≤ 46.00
- IHD ≤ 0.48

## Conclusions

The decision tree model provides actionable predictions for profitability, highlighting key factors and offering strategic recommendations. The analysis will aid the company in navigating challenges and capitalizing on growth opportunities.

