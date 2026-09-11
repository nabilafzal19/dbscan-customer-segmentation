# DBSCAN Customer Segmentation

A production-oriented customer segmentation project using **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**. The project covers data preprocessing, feature scaling, DBSCAN hyperparameter tuning, cluster evaluation, business interpretation, production artifact creation, MLflow experiment tracking, and FastAPI inference.

---

## Project Overview

The objective is to segment customers based on:

* Age
* Annual Income
* Spending Score

Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand and can identify customers that do not belong to sufficiently dense regions as **noise**.

### Machine Learning Pipeline

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
StandardScaler
       ↓
DBSCAN
       ↓
Hyperparameter Tuning
       ↓
Cluster Evaluation
       ↓
Business Interpretation
       ↓
Production Artifact
       ↓
MLflow Tracking
       ↓
FastAPI
```

---

## Dataset

The project uses the **Mall Customers Dataset**.

Dataset source:

```text
https://raw.githubusercontent.com/sharmaroshan/Clustering-of-Mall-Customers/master/Mall_Customers.csv
```

The dataset contains **200 customers** and the following columns:

| Column                 | Description                           |
| ---------------------- | ------------------------------------- |
| CustomerID             | Unique customer identifier            |
| Genre                  | Customer gender                       |
| Age                    | Customer age                          |
| Annual Income (k$)     | Annual income in thousands of dollars |
| Spending Score (1-100) | Customer spending score               |

### Example Dataset

| CustomerID | Genre  | Age | Annual Income (k$) | Spending Score (1-100) |
| ---------: | ------ | --: | -----------------: | ---------------------: |
|          1 | Male   |  19 |                 15 |                     39 |
|          2 | Male   |  21 |                 15 |                     81 |
|          3 | Female |  20 |                 16 |                      6 |
|          4 | Female |  23 |                 16 |                     77 |
|          5 | Female |  31 |                 17 |                     40 |
|          6 | Female |  22 |                 17 |                     76 |
|          7 | Female |  35 |                 18 |                      6 |
|          8 | Female |  23 |                 18 |                     94 |
|          9 | Male   |  64 |                 19 |                      3 |
|         10 | Female |  30 |                 19 |                     72 |

---

## Feature Selection

The following features were used for DBSCAN:

```text
Age
Annual Income (k$)
Spending Score (1-100)
```

### Features excluded

**CustomerID**

CustomerID is an identifier and does not contain meaningful behavioral information.

**Genre**

Genre was excluded from the initial DBSCAN model because directly encoding it as `0/1` and including it in Euclidean distance would introduce an arbitrary numerical distance into the clustering process.

---

## Data Preprocessing

The selected numerical features were standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Scaling is important because DBSCAN uses distance to determine neighborhood density.

Without scaling, features with larger numerical ranges can disproportionately influence the distance calculation.

---

# DBSCAN

DBSCAN is a density-based clustering algorithm.

It uses two main parameters:

### `eps`

Defines the maximum distance within which points are considered neighbors.

### `min_samples`

Defines the minimum number of samples required within the neighborhood for a point to be considered a core point.

---

## DBSCAN Point Types

### Core Point

A point located in a sufficiently dense region.

### Border Point

A point that is not dense enough to be a core point itself but is reachable from a core point.

### Noise

A point that does not belong to any sufficiently dense region.

DBSCAN represents noise using:

```text
-1
```

---

# Hyperparameter Selection

A k-distance graph was used to help investigate appropriate `eps` values.

Multiple combinations of `eps` and `min_samples` were evaluated using:

* Number of clusters
* Number of noise points
* Noise percentage
* Silhouette score
* Cluster visualization
* Business usefulness

### Selected Configuration

```text
Algorithm      = DBSCAN
eps            = 0.55
min_samples    = 8
```

---

# Final Model Results

```text
Total Customers = 200
Clusters        = 5
Core Points     = 80
Border Points   = 58
Noise Points    = 62
Noise Percentage = 31%
Silhouette Score = 0.5158
```

### Point Distribution

| Point Type | Customers | Percentage |
| ---------- | --------: | ---------: |
| Core       |        80 |        40% |
| Border     |        58 |        29% |
| Noise      |        62 |        31% |
| **Total**  |   **200** |   **100%** |

---

# Customer Segments

## Cluster 0 — Young High-Spenders

| Metric         | Average |
| -------------- | ------: |
| Customers      |      19 |
| Age            |   23.74 |
| Annual Income  |  26.11k |
| Spending Score |   78.32 |

Characteristics:

* Young customers
* Lower annual income
* High spending score

Potential business strategy:

* Youth-focused campaigns
* Discounts
* Social media campaigns
* High-engagement offers

---

## Cluster 1 — Mature Regular Customers

| Metric         | Average |
| -------------- | ------: |
| Customers      |      49 |
| Age            |   54.63 |
| Annual Income  |  54.04k |
| Spending Score |   48.59 |

Characteristics:

* Older customers
* Medium income
* Average spending behavior

Potential business strategy:

* Regular promotions
* Loyalty programs
* Personalized recommendations

---

## Cluster 2 — Young Potential Customers

| Metric         | Average |
| -------------- | ------: |
| Customers      |      29 |
| Age            |   24.38 |
| Annual Income  |  54.52k |
| Spending Score |   50.17 |

Characteristics:

* Young customers
* Medium income
* Average spending

Potential business strategy:

* Engagement campaigns
* Personalized recommendations
* Cross-selling
* Loyalty incentives

---

## Cluster 3 — Premium High-Value Customers

| Metric         | Average |
| -------------- | ------: |
| Customers      |      33 |
| Age            |   32.73 |
| Annual Income  |  81.06k |
| Spending Score |   83.00 |

Characteristics:

* Higher income
* High spending
* Strong commercial value

Potential business strategy:

* Premium products
* VIP programs
* Personalized offers
* Loyalty rewards

---

## Cluster 4 — High-Income Low-Spenders

| Metric         | Average |
| -------------- | ------: |
| Customers      |       8 |
| Age            |   46.12 |
| Annual Income  |  79.62k |
| Spending Score |   16.00 |

Characteristics:

* High income
* Very low spending

Potential business strategy:

* Re-engagement campaigns
* Personalized promotions
* Product recommendations
* Customer research

---

# Core / Border Distribution

| Cluster | Core | Border | Total |
| ------- | ---: | -----: | ----: |
| 0       |    7 |     12 |    19 |
| 1       |   33 |     16 |    49 |
| 2       |   17 |     12 |    29 |
| 3       |   22 |     11 |    33 |
| 4       |    1 |      7 |     8 |
| Noise   |    0 |      0 |    62 |

Cluster 1 and Cluster 3 have strong core populations, indicating relatively dense customer regions.

Cluster 4 contains only one core point and seven border points, making it a relatively weak density-based cluster.

---

# Noise Analysis

DBSCAN classified **62 customers (31%)** as noise.

Noise does not automatically mean that these customers are business anomalies.

It means that, under the selected DBSCAN parameters, these customers do not belong to sufficiently dense regions.

The noise population had:

| Metric                 |  Value |
| ---------------------- | -----: |
| Customers              |     62 |
| Average Age            |   40.2 |
| Average Income         | 66.24k |
| Average Spending Score |  29.07 |

Examples of unusual profiles include:

```text
Income = 137k, Spending Score = 18
Income = 120k, Spending Score = 16
Income = 137k, Spending Score = 83
Income = 15k, Spending Score = 99
```

The noise population can therefore be treated as a separate population for further analysis rather than automatically forcing every customer into a cluster.

---

# DBSCAN vs K-Means

| Property                 | K-Means                              | DBSCAN                                |
| ------------------------ | ------------------------------------ | ------------------------------------- |
| Type                     | Centroid-based                       | Density-based                         |
| Number of clusters       | Must specify K                       | Does not require K                    |
| Handles noise            | No explicit noise class              | Yes                                   |
| Outlier sensitivity      | More sensitive                       | Can isolate sparse points             |
| Cluster shape            | Best for relatively compact clusters | Can identify density-based structures |
| Prediction of new points | Native `predict()`                   | No native `predict()` in sklearn      |
| Main parameters          | `n_clusters`                         | `eps`, `min_samples`                  |

For this dataset, K-Means is useful when every customer must receive a segment.

DBSCAN is useful when we want to discover dense behavioral groups and identify customers that do not naturally belong to those groups.

---

# Production Inference

Scikit-learn's DBSCAN implementation does not provide a native `predict()` method for unseen customers.

Therefore, this project implements a custom inference strategy.

### Inference Flow

```text
New Customer
     ↓
StandardScaler
     ↓
Find Nearest DBSCAN Core Point
     ↓
Calculate Distance
     ↓
Distance <= eps?
     │
 ┌───┴────┐
 YES      NO
  ↓        ↓
Cluster   Noise
```

The new customer is assigned to the cluster of the nearest DBSCAN core point if that core point is within:

```text
eps = 0.55
```

Otherwise:

```text
cluster = -1
```

The API also returns the distance to the nearest core point.

---

# Production Artifact

The complete inference artifact is stored as:

```text
model/dbscan_customer_segmentation.joblib
```

The artifact contains:

```text
scaler
core_neighbors
core_labels
eps
min_samples
features
cluster_names
```

This allows the production application to perform inference without retraining the model.

---

# MLflow

MLflow is used for experiment tracking.

Tracked parameters include:

```text
algorithm
eps
min_samples
features
```

Tracked metrics include:

```text
n_clusters
core_points
border_points
noise_points
noise_percentage
silhouette_score
```

Artifacts include:

```text
DBSCAN production artifact
DBSCAN tuning results
```

---

# FastAPI

The project exposes a REST API for customer segmentation.

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "DBSCAN",
  "eps": 0.55,
  "min_samples": 8
}
```

### Prediction

```text
POST /predict
```

Example request:

```json
{
  "age": 30,
  "annual_income": 80,
  "spending_score": 90
}
```

Example response:

```json
{
  "cluster": 3,
  "segment": "Premium High-Value Customer",
  "distance_to_nearest_core": 0.31
}
```

---

# Project Structure

```text
dbscan-customer-segmentation/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── predictor.py
│
├── model/
│   └── dbscan_customer_segmentation.joblib
│
├── tests/
│   └── test_api.py
│
├── notebooks/
│   └── 03_dbscan_customer_segmentation.ipynb
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* DBSCAN
* StandardScaler
* NearestNeighbors
* Joblib
* MLflow
* FastAPI
* Pydantic
* Uvicorn
* Docker
* Pytest

---

# Running Locally

## Install dependencies

```bash
pip install -r requirements.txt
```

## Start the API

```bash
uvicorn app.main:app --reload
```

## API Documentation

Open:


```text
http://localhost:8000/docs
```

---

# Key ML Engineering Learnings

This project demonstrates:

* Density-based clustering
* DBSCAN fundamentals
* `eps` selection
* `min_samples` selection
* k-distance analysis
* Core, border and noise points
* Silhouette-based evaluation
* Hyperparameter tuning
* Business-oriented cluster interpretation
* Handling clustering noise
* Feature scaling
* Production model artifacts
* Custom inference for DBSCAN
* MLflow experiment tracking
* FastAPI model serving
* Docker-based deployment

---

# Final Conclusion

DBSCAN identified **five meaningful density-based customer segments** while also identifying **31% of customers as noise**.

The final configuration was:

```text
DBSCAN
eps = 0.55
min_samples = 8
```

with:

```text
5 clusters
80 core points
58 border points
62 noise points
31% noise
0.5158 silhouette score
```

The model demonstrates that customer behavior does not necessarily form perfectly separated groups. DBSCAN is therefore useful not only for segmentation but also for identifying customers whose behavior does not strongly match a dense customer population.

For a business requiring every customer to receive a segment, DBSCAN should be evaluated against alternative approaches such as K-Means or a hybrid segmentation strategy rather than being treated as the sole assignment model.
