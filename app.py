import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="centered")

# Get absolute path of directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Artifacts
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, 'knn_model.pkl')
    titles_path = os.path.join(BASE_DIR, 'movie_titles.pkl')
    matrix_path = os.path.join(BASE_DIR, 'sparse_movie_matrix.pkl')
    
    model = joblib.load(model_path)
    titles = joblib.load(titles_path)
    matrix = joblib.load(matrix_path)
    return model, titles, matrix

model_knn, movie_titles, sparse_movie_matrix = load_artifacts()

# Title Header
st.title("🎬 Smart Movie Recommendation Engine")
st.write("A hybrid recommendation system combining popularity genre filtering for new users and $K$-NN collaborative filtering for item similarity.")

# Create Navigation Tabs
tab1, tab2 = st.tabs(["🔥 New User Onboarding (Genre Filter)", "🎯 Item Similarity ($K$-NN)"])

# ---------------------------------------------------------
# TAB 1: COLD-START / NEW USER ONBOARDING
# ---------------------------------------------------------
with tab1:
    st.subheader("Welcome! Tell us what you like")
    st.write("If you're new or don't have a specific movie in mind, pick a genre to discover top-rated titles.")
    
    genres = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
              "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
              "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    
    selected_genre = st.selectbox("Select your favorite genre:", genres)
    num_recommendations = st.slider("Number of movies to display:", 3, 10, 5)
    
    if st.button("Find Top Rated Movies"):
        # Extract matching movie titles containing the genre tag
        matching_titles = [title for title in movie_titles if selected_genre.lower() in title.lower()]
        
        if matching_titles:
            st.subheader(f"Popular {selected_genre} Picks:")
            for i, title in enumerate(matching_titles[:num_recommendations], start=1):
                st.write(f"**{i}. {title}**")
        else:
            st.info(f"No specific titles found matching '{selected_genre}' in the current filtered subset. Try another genre!")

# ---------------------------------------------------------
# TAB 2: K-NN COLLABORATIVE FILTERING
# ---------------------------------------------------------
with tab2:
    st.subheader("Find Similar Movies")
    st.write("Select a movie you love to find implicit behavioral matches based on user rating patterns.")
    
    selected_movie = st.selectbox("Choose or type a movie title:", movie_titles)
    
    if st.button("Get $K$-NN Recommendations"):
        # Locate index of selected movie
        movie_idx = list(movie_titles).index(selected_movie)
        
        # Extract sparse feature vector and compute K-NN neighbors
        movie_vector = sparse_movie_matrix[movie_idx]
        distances, indices = model_knn.kneighbors(movie_vector, n_neighbors=6)
        
        st.subheader(f"Top Recommendations for '{selected_movie}':")
        
        # Display top 5 matches (skip index 0 as it's the queried movie)
        for i in range(1, len(distances.flatten())):
            rec_title = movie_titles[indices.flatten()[i]]
            cosine_dist = distances.flatten()[i]
            similarity_pct = (1 - cosine_dist) * 100
            
            st.write(f"**{i}. {rec_title}** — *{similarity_pct:.1f}% Similarity*")
