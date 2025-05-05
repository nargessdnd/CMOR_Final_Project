# utils/data_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set default plot styles
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

def plot_esg_distributions(df):
    """Plots distributions of ESG scores."""
    esg_cols = [col for col in df.columns if 'ESG' in col]
    if not esg_cols:
        print("No ESG columns found for plotting distributions.")
        return
    print("Plotting ESG Score Distributions...")
    n_cols = 3
    n_rows = (len(esg_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for i, col in enumerate(esg_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

def plot_financial_metrics(df):
    """Plots distributions of key financial metrics."""
    financial_cols = ['Revenue', 'ProfitMargin', 'MarketCap', 'GrowthRate'] # Adjust if needed
    plot_cols = [col for col in financial_cols if col in df.columns]
    if not plot_cols:
        print("No financial columns found for plotting distributions.")
        return
    print("Plotting Financial Metric Distributions...")
    n_cols = 2
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 6, n_rows * 5))
    for i, col in enumerate(plot_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col].dropna(), kde=True) # Drop NA for plotting
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, method='pearson'):
    """Plots the correlation matrix of numerical features."""
    numerical_df = df.select_dtypes(include=np.number)
    if numerical_df.empty:
        print("No numerical columns found for correlation matrix.")
        return
    print(f"Plotting Correlation Matrix (Method: {method})...")
    corr = numerical_df.corr(method=method)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

def plot_esg_vs_financial(df, esg_col='ESG_Overall', financial_col='ProfitMargin'):
    """Plots a scatter plot of a specific ESG score vs. a financial metric."""
    if esg_col not in df.columns or financial_col not in df.columns:
        print(f"Error: Columns '{esg_col}' or '{financial_col}' not found.")
        return
    print(f"Plotting Scatter Plot: {esg_col} vs {financial_col}...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=esg_col, y=financial_col, alpha=0.6)
    plt.title(f'{esg_col} vs. {financial_col}')
    plt.xlabel(esg_col)
    plt.ylabel(financial_col)
    plt.grid(True)
    plt.show()

def plot_pca_results(pca):
    """Plots the cumulative explained variance ratio for PCA."""
    print("Plotting PCA Explained Variance Ratio...")
    plt.figure(figsize=(8, 5))
    explained_variance_ratio = getattr(pca, 'explained_variance_ratio_', None)
    if explained_variance_ratio is None:
        print("PCA object does not have 'explained_variance_ratio_'. Cannot plot.")
        return

    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA: Explained Variance vs. Number of Components')
    plt.grid(True)
    # Add a line for 95% variance threshold
    plt.axhline(y=0.95, color='r', linestyle=':', label='95% Variance Threshold')
    plt.legend(loc='best')
    plt.show()

def plot_clusters(X, labels, centroids=None, title='Cluster Visualization'):
    """
    Plots clustering results, typically using the first two dimensions of X.
    Assumes X is 2D (e.g., after PCA).
    """
    print(f"Plotting Clusters: {title}...")
    plt.figure(figsize=(10, 7))
    # Use a colormap suitable for categorical data
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    for k, col in zip(unique_labels, colors(range(len(unique_labels)))):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6, alpha=0.6, label=f'Cluster {k}' if k!=-1 else 'Noise')

    if centroids is not None:
        # Ensure centroids are 2D for plotting
        if centroids.shape[1] >= 2:
             plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroids', edgecolors='black')
        else:
             print("Warning: Centroids are not 2D, cannot plot them.")


    plt.title(title)
    plt.xlabel('Feature 1 (e.g., PC1)')
    plt.ylabel('Feature 2 (e.g., PC2)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# --- NEWLY ADDED FUNCTIONS ---

def plot_regression_results(y_true, y_pred, title='Regression Results'):
    """
    Plots actual vs. predicted values for regression tasks.
    """
    print(f"Plotting Regression Results: {title}...")
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', label='Predictions')
    # Plotting the perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure x and y axes have the same scale
    plt.show()

def plot_classification_results(y_true, y_pred, title='Classification Results', history=None):
    """
    Plots confusion matrix for classification tasks.
    Optionally plots learning history (e.g., loss or error over iterations) if provided.
    """
    print(f"Plotting Classification Results: {title}...")
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(1, 2 if history is not None else 1, figsize=(12 if history is not None else 6, 5))

    if history is not None:
        # Plot Learning Curve
        ax[0].plot(history)
        ax[0].set_title(f'{title} - Learning Curve (e.g., Error/Loss)')
        ax[0].set_xlabel('Iteration / Epoch')
        ax[0].set_ylabel('Error / Loss')
        ax[0].grid(True)

        # Plot Confusion Matrix on the second subplot
        disp.plot(ax=ax[1], cmap='Blues')
        ax[1].set_title(f'{title} - Confusion Matrix')
    else:
        # Plot Confusion Matrix on the single subplot
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'{title} - Confusion Matrix')

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_dict, title='Model Comparison', metric='MSE', higher_is_better=False):
    """
    Compares model performance based on a specific metric using a bar plot.

    Args:
        results_dict (dict): A dictionary where keys are model names and values are
                             dictionaries containing metric names and scores.
                             Example: {'Linear Regression': {'MSE': 0.5, 'R2': 0.8}, ...}
        title (str): The title for the plot.
        metric (str): The metric to compare (e.g., 'MSE', 'Accuracy', 'F1', 'Silhouette').
        higher_is_better (bool): Whether a higher value of the metric indicates better performance.
    """
    print(f"Plotting Model Comparison: {title} (Metric: {metric})")
    model_names = list(results_dict.keys())
    metric_values = [results_dict[model].get(metric, np.nan) for model in model_names] # Use NaN if metric missing

    # Remove models where the metric is NaN
    valid_indices = [i for i, v in enumerate(metric_values) if not np.isnan(v)]
    if not valid_indices:
        print(f"No valid data found for metric '{metric}'. Cannot plot comparison.")
        return
    model_names = [model_names[i] for i in valid_indices]
    metric_values = [metric_values[i] for i in valid_indices]


    plt.figure(figsize=(max(10, len(model_names) * 1.5), 6)) # Adjust width based on number of models
    bars = plt.bar(model_names, metric_values, color=sns.color_palette("viridis", len(model_names)))
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability if many models

    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom' if higher_is_better else 'top', ha='center') # Adjust vertical alignment

    # Adjust y-axis limits slightly for better visualization
    if metric_values:
       min_val = min(metric_values)
       max_val = max(metric_values)
       padding = (max_val - min_val) * 0.1
       if higher_is_better:
           plt.ylim(max(0, min_val - padding), max_val + padding) # Ensure y-axis starts near 0 if applicable
       else:
           plt.ylim(min_val - padding, max_val + padding)


    plt.grid(axis='y', linestyle='--')
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()