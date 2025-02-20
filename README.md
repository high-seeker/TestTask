# Machine Learning Model for Approximating QW and DP

This project focuses on building a machine learning model to approximate and generalize the target variables `QW` and `DP` based on the features `s_mt`, `s_mq`, `d`, and `h_p`. The dataset is provided in the file `2022_Test_ML.csv`.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)

---

## Project Overview

The goal of this project is to:
1. Analyze the dataset to understand its properties and quality.
2. Build a regression model to approximate the target variables `QW` and `DP`.
3. Evaluate the model's performance using the coefficient of determination (R²).
4. Visualize the dependency of model errors on the dataset size.

The chosen model is a **Random Forest Regressor**, which is robust, handles non-linear relationships, and can model multiple targets.

---

## Dataset Description

The dataset consists of:
- **Features:**
  - `s_mt`: Range [0.8, 2.7]
  - `s_mq`: Range [0.8, 2.1]
  - `d`: Range [1.0, 3.0]
  - `h_p`: Range [4.0, 10.0]
- **Targets:**
  - `QW`: Target variable 1
  - `DP`: Target variable 2

The dataset is stored in the file `2022_Test_ML.csv`.

---

## Requirements

To run this project, you need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
Installation
Clone the repository:

bash
git clone https://github.com/high-seeker/TestTask.git
Navigate to the project directory:

bash
cd your-repo-name
Install the required libraries (see Requirements).

Usage
Open the Jupyter Notebook:

bash
jupyter notebook
Run the notebook ML_Model_Building.ipynb step by step.

The notebook includes:

Data loading and cleaning.

Exploratory data analysis (EDA).

Model training and evaluation.

Visualization of results.

Results
Model Performance
R² (Train): 0.95 (example)

R² (Test): 0.92 (example)

Visualizations
Pairplot: Shows the distribution of features and their relationships.

Correlation Matrix: Highlights correlations between features and targets.

R² vs. Dataset Size: Demonstrates how model performance improves with more data.

Acknowledgments
Thanks to the creators of pandas, numpy, matplotlib, seaborn, and scikit-learn for their amazing libraries.
