import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="centered")

st.title("🎬 K-NN Movie Recommendation Engine")
st.write("Select a movie to discover top similar movies using Item-Based Collaborative Filtering.")

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('knn_model.pkl')
    titles = joblib.load('movie_titles.pkl')
    matrix = joblib.load('sparse_movie_matrix.pkl')
    return model, titles, matrix

model_knn, movie_titles, sparse_movie_matrix = load_artifacts()

# Dropdown menu for selecting movies
selected_movie = st.selectbox("Choose or type a movie name:", movie_titles)

# Recommendation button
if st.button("Get Recommendations"):
    # Find index of selected movie
    movie_idx = list(movie_titles).index(selected_movie)
    
    # Extract feature vector and query KNN
    movie_vector = sparse_movie_matrix[movie_idx]
    distances, indices = model_knn.kneighbors(movie_vector, n_neighbors=6)
    
    st.subheader(f"Top 5 Recommendations for '{selected_movie}':")
    
    # Display results (skipping the 0-th index because it is the query movie itself)
    for i in range(1, len(distances.flatten())):
        rec_title = movie_titles[indices.flatten()[i]]
        cosine_dist = distances.flatten()[i]
        similarity_pct = (1 - cosine_dist) * 100
        
        st.write(f"**{i}. {rec_title}** — *{similarity_pct:.1f}% Similarity*")
