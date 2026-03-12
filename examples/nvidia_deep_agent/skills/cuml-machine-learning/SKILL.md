---
name: cuml-machine-learning
description: Use for GPU-accelerated machine learning on tabular data using NVIDIA cuML. Triggers when tasks involve classification, regression, clustering, dimensionality reduction, or model training on datasets.
---

# cuML Machine Learning Skill

GPU-accelerated machine learning using NVIDIA RAPIDS cuML. cuML provides a scikit-learn-compatible API that runs on NVIDIA GPUs, enabling massive speedups on large datasets.

## When to Use This Skill

Use this skill when:
- Training classification models (predict categories, detect fraud, classify text)
- Training regression models (forecast values, predict prices, estimate quantities)
- Clustering data (segment customers, group documents, find patterns)
- Dimensionality reduction (visualize high-dimensional data, compress features)
- Preprocessing and feature engineering on large datasets
- Any ML task on datasets with 10K+ rows where GPU acceleration helps

## Initialization (REQUIRED)

Always start every script with this boilerplate. It tests actual GPU ML operations.

```python
import pandas as pd
import numpy as np

try:
    import cudf
    import cuml
    # Smoke-test: verify GPU ML works end-to-end
    _test_data = cudf.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0, 8.0]})
    _km = cuml.cluster.KMeans(n_clusters=2, n_init=1, random_state=42)
    _km.fit(_test_data)
    assert len(_km.labels_) == 4
    GPU = True
except Exception as e:
    print(f"[GPU] cuml unavailable, falling back to scikit-learn: {e}")
    GPU = False

def read_csv(path):
    return cudf.read_csv(path) if GPU else pd.read_csv(path)

def to_pd(df):
    """Convert cuML/cuDF output to pandas. Use this instead of .to_pandas() directly."""
    if not GPU:
        return df
    try:
        return df.to_pandas()
    except Exception as e:
        print(f"[GPU] .to_pandas() failed, using Arrow fallback: {e}")
        return df.to_arrow().to_pandas()
```

## Import Patterns

```python
# GPU mode
if GPU:
    from cuml.cluster import KMeans, DBSCAN, HDBSCAN
    from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
    from cuml.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
    from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from cuml.svm import SVC, SVR
    from cuml.decomposition import PCA, TruncatedSVD
    from cuml.manifold import UMAP, TSNE
    from cuml.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from cuml.model_selection import train_test_split
    from cuml.metrics import accuracy_score, r2_score, mean_squared_error
# CPU fallback
else:
    from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    # UMAP not in sklearn — skip or pip install umap-learn
```

## Quick Reference

### Train/Test Split (Start Here)

```python
X = df[["feature1", "feature2", "feature3"]].astype("float32")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Classification

```python
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = float(accuracy_score(to_pd(y_test), to_pd(predictions)))
print(f"Accuracy: {accuracy:.4f}")

# Feature importances (tree models only)
importances = to_pd(model.feature_importances_)
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")
```

### Regression

```python
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
r2 = float(r2_score(to_pd(y_test), to_pd(predictions)))
mse = float(mean_squared_error(to_pd(y_test), to_pd(predictions)))
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# Coefficients
coeffs = to_pd(model.coef_)
print(f"Intercept: {float(model.intercept_):.4f}")
```

### Clustering (KMeans)

```python
X = df[["feature1", "feature2"]].astype("float32")

model = KMeans(n_clusters=4, n_init=10, random_state=42)
model.fit(X)

labels = to_pd(model.labels_)
centroids = to_pd(model.cluster_centers_)
inertia = float(model.inertia_)

print(f"Inertia: {inertia:.2f}")
print(f"Cluster sizes: {labels.value_counts().sort_index().to_dict()}")
print(f"Centroids:\n{centroids}")
```

### Dimensionality Reduction (PCA)

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype("float32"))

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_scaled)

variance_ratio = to_pd(pca.explained_variance_ratio_)
print(f"Explained variance: {[f'{v:.4f}' for v in variance_ratio]}")
print(f"Total explained: {float(sum(variance_ratio)):.4f}")
```

### Dimensionality Reduction (UMAP — GPU only)

```python
if GPU:
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = to_pd(reducer.fit_transform(X_scaled))
    print(f"UMAP embedding shape: {embedding.shape}")
```

### Preprocessing

```python
# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype("float32"))

# Encode categorical columns
le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["category"])
```

## Data Type Requirements

- cuML requires **float32 or float64** for features. Always cast: `X.astype("float32")`
- Integer targets (classification labels) work directly
- Categorical columns must be encoded first (LabelEncoder or OneHotEncoder)
- cuML does NOT support sparse matrices — always use dense data

## Gotchas

| Issue | Fix |
|-------|-----|
| `TypeError: sparse input` | Convert to dense: `X.toarray()` or don't use sparse |
| PCA `solver='randomized'` fails | Use `solver='full'` or omit (cuML auto-selects) |
| UMAP not available on CPU | Skip UMAP in CPU mode or `pip install umap-learn` |
| Float64 slower than float32 | Cast to float32: `X.astype("float32")` |
| Large dataset OOM | Reduce features or sample data before fitting |

## Output Guidelines

When reporting ML results:
- Include dataset shape (rows × features) and target distribution
- Show train/test split sizes
- Report key metrics in a formatted table (accuracy, R², MSE, etc.)
- For classification: show per-class metrics if multi-class
- For clustering: show cluster sizes and centroid summaries
- For dimensionality reduction: show explained variance ratios
- List feature importances ranked by magnitude
- Note any data quality issues (class imbalance, missing values, outliers)
