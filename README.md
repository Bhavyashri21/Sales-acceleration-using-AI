# Sales Acceleration using AI

## 1. Introduction
This project presents an AI-based approach for predictive sales pipeline analytics. The objective is to estimate the probability of a sales lead being successfully converted (won) using machine learning techniques and temporal interaction modeling. The implementation is inspired by research on Hawkes process-based modeling of sales activities.

Traditional sales forecasting methods rely heavily on subjective judgments by sales representatives, which often suffer from bias and inconsistency. This project replaces subjective estimation with a data-driven approach.

---

## 2. Objectives
- To develop a system that predicts the likelihood of converting a sales lead.
- To analyze the effect of temporal interaction patterns on sales outcomes.
- To compare different models for prediction accuracy.
- To provide an interactive interface for real-time lead evaluation.

---

## 3. Methodology

### 3.1 Data Generation
Synthetic sales data is generated to simulate real-world sales pipelines. Each lead contains:
- Profile features (geography, deal size, sector, product line)
- Interaction timestamps
- Outcome (won/lost)

Won leads exhibit clustered interaction patterns, while lost leads show sparse interactions.

### 3.2 Feature Engineering
Key features extracted include:
- Number of interactions
- Recency of last interaction
- Mean gap between interactions
- Clustering score of interactions
- Lead age and sales stage

Categorical variables are encoded into numerical form and standardized.

### 3.3 Models Used
- Baseline Model (Random scoring)
- Logistic Regression
- Hawkes Process Model (self-exciting temporal model)

The Hawkes model captures the temporal clustering of interactions, which is a key indicator of sales success.

---

## 4. Implementation

The project is implemented in Python and organized into modular components:

- `data_generator.py`: Generates synthetic sales data
- `hawkes_model.py`: Implements the Hawkes process model
- `train_evaluate.py`: Trains models and evaluates performance
- `dashboard.py`: Provides an interactive interface for scoring leads
- `requirements.txt`: Lists required dependencies

---

## 5. Results

The performance of different models is evaluated using the Area Under the ROC Curve (AUC):

- Baseline Model: ~0.50
- Logistic Regression: ~0.65–0.72
- Hawkes Model: ~0.70–0.78

The Hawkes model demonstrates improved performance due to its ability to capture temporal interaction patterns.

---

## 6. How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
python data_generator.py
python train_evaluate.py
python dashboard.py
```


---

## 7. Conclusion
This project demonstrates the effectiveness of machine learning in improving sales pipeline forecasting by replacing subjective assessments with objective, data-driven methods. 

The Hawkes process model, in particular, provides a significant advantage by capturing temporal interaction patterns between sales representatives and leads. These temporal dynamics serve as strong indicators of conversion likelihood, resulting in improved predictive performance compared to traditional models.

Overall, the system highlights the potential of AI-driven approaches in enhancing decision-making and efficiency in sales management.

---
