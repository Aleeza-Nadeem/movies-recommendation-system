# 🎬 Movie Recommender System using K-Nearest Neighbors (K-NN)

An Item-Based Collaborative Filtering recommendation engine built with Python, Scikit-Learn, and Streamlit. The system measures spatial distance between movie rating vectors across high-dimensional user interaction space to generate personalized recommendations.

---

## 📌 Project Overview

This project implements an **Item-Based Collaborative Filtering** approach using the K-Nearest Neighbors algorithm. Instead of relying on manual genre metadata, the system constructs a 2D coordinate grid where:
* **Rows (Data Points):** Unique movie titles.
* **Columns (Dimensions):** Individual user IDs.
* **Values (Coordinates):** Numerical user ratings (1.0 to 5.0).

By modeling rating habits as multi-dimensional spatial vectors, K-NN calculates the relative angle between movies to identify similar titles enjoyed by the same user communities.

---

## 🛠️ Key Features & Technical Decisions

* **Collaborative Filtering:** Uses real user interaction patterns rather than explicit genre metadata to discover implicit cross-genre similarities.
* **Sparse Matrix Optimization:** Converts dense DataFrames into SciPy `csr_matrix` representations, discarding unrated zero-cells to optimize memory footprint and computation speed.
* **Noise Reduction:** Filters out sparse items with fewer than 7 total ratings to remove statistical outliers and cold-start items.
* **Cosine Distance Metric:** Uses vector angle calculation instead of Euclidean distance to eliminate user rating scale bias without needing standard feature scaling.

---

## 📊 Dataset & Pipeline

1. **Preprocessing & Merging:** Merged MovieLens datasets (`movies.csv` and `ratings.csv`) on `movieId`.
2. **Item Thresholding:** Applied a minimum frequency threshold ($\ge 7$ ratings per title) to prune unrated noise.
3. **Pivoting:** Transformed long-format interaction logs into a 2D User-Item grid matrix (`pivot_table`).
4. **Compression:** Encoded the pivot grid into a Compressed Sparse Row (`csr_matrix`) format.
5. **Model Fitting:** Instantiated `sklearn.neighbors.NearestNeighbors` using `metric='cosine'` and `algorithm='brute'`.

---

## 🚀 Interactive Streamlit Web App

The repository includes a ready-to-deploy Streamlit application (`app.py`).

### File Structure
```text
├── app.py                   # Streamlit web application
├── requirements.txt         # Production dependencies
├── knn_model.pkl            # Fitted K-NN model artifact
├── movie_titles.pkl         # Array of movie grid title indices
├── sparse_movie_matrix.pkl  # Compressed SciPy CSR sparse matrix
└── README.md                # Project documentation