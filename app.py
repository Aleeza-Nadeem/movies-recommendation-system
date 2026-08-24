import os
import joblib
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="centered")

# Get absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved artifacts using robust relative paths
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, 'knn_model.pkl')
    titles_path = os.path.join(BASE_DIR, 'movie_titles.pkl')
    matrix_path = os.path.join(BASE_DIR, 'sparse_movie_matrix.pkl')
    
    model = joblib.load(model_path)
    titles = joblib.load(titles_path)
    matrix = joblib.load(matrix_path)
    return model, titles, matrix

# Load model and data
model_knn, movie_titles, sparse_movie_matrix = load_artifacts()

# App UI
st.title("🎬 K-NN Movie Recommendation Engine")
st.write("Select a movie to discover top similar movies using Item-Based Collaborative Filtering.")

# Dropdown menu for movie selection
selected_movie = st.selectbox("Choose or type a movie name:", movie_titles)

# Recommendation Trigger
if st.button("Get Recommendations"):
    # Locate movie index in grid
    movie_idx = list(movie_titles).index(selected_movie)
    
    # Extract feature vector and query KNN
    movie_vector = sparse_movie_matrix[movie_idx]
    distances, indices = model_knn.kneighbors(movie_vector, n_neighbors=6)
    
    st.subheader(f"Top 5 Recommendations for '{selected_movie}':")
    
    # Display top 5 recommendations (skip index 0 as it is the queried movie itself)
    for i in range(1, len(distances.flatten())):
        rec_title = movie_titles[indices.flatten()[i]]
        cosine_dist = distances.flatten()[i]
        similarity_pct = (1 - cosine_dist) * 100
        
        st.write(f"**{i}. {rec_title}** — *{similarity_pct:.1f}% Similarity*")
