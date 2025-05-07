# CMOR 438 / INDE 577: Final Project - Data Science & Machine Learning Showcase

**Author:** Narges Saeednejad (ns112)

## Project Overview

This repository demonstrates the implementation and application of key supervised and unsupervised machine learning algorithms. The project focuses on analyzing ESG (Environmental, Social, and Governance) metrics and their relationship with financial performance using custom implementations of machine learning algorithms.

## Repository Structure

```
project/
├── .gitignore
├── LICENSE
├── README.md
│
├── Supervised_Learning/
│   ├── __init__.py
│   ├── perceptron.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── neural_network.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── ensemble_methods.py
│   └── README.md
│
├── Unsupervised_Learning/
│   ├── __init__.py
│   ├── kmeans.py
│   ├── dbscan.py
│   ├── pca.py
│   ├── svd_compression.py
│   ├── label_propagation.py
│   └── README.md
│
├── utils/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── data_visualization.py
│
├── data/
│   ├── company_esg_financial_dataset.csv
│   └── cifar10/ (for image processing examples)
│
└── ESG_Financial_Analysis.ipynb
```

## Setup Instructions

### Requirements
- Python 3.7+
- Key dependencies:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - jupyter

### Installation
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install numpy pandas matplotlib scikit-learn jupyter`

## Running the Analysis

1. Start Jupyter notebook: `jupyter notebook`
2. Open `ESG_Financial_Analysis.ipynb`
3. Run the cells to see the analysis and comparison of different machine learning techniques applied to ESG and financial data

The notebook demonstrates the practical application of the custom-implemented algorithms in both supervised and unsupervised learning directories. 