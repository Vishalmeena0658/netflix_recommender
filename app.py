import streamlit as st
import pandas as pd
from configs.config import Config
from src.data.loader import DatasetLoader
from src.models.recommenders import PureSVDRecommender, ContentBasedRecommender, HybridRecommender

st.set_page_config(page_title="Netflix RecSys", layout="wide")
st.title("🎬 Netflix Recommendation System Dashboard")

@st.cache_resource
def load_and_train():
    df, movies = DatasetLoader.load_dataset()
    svd, cb = PureSVDRecommender(), ContentBasedRecommender()
    svd.fit(df.head(100000)); cb.fit(df.head(100000), movies)
    return df, movies, HybridRecommender(svd, cb, movies)

df, movies_df, hybrid = load_and_train()

user_id = st.number_input("Enter User ID", min_value=1, value=int(df['user_id'].iloc[0]))
if st.button("Generate Recommendations"):
    st.subheader(f"Top Recommendations for User {user_id}")
    
    movie_scores = []
    for mid in movies_df['movie_id'].unique()[:500]: # Sample catalog limit
        pred = hybrid.predict(user_id, mid)
        movie_scores.append((mid, pred))
        
    top_recs = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:5]
    
    for mid, score in top_recs:
        title = movies_df[movies_df['movie_id'] == mid]['title'].values[0]
        explanation = hybrid.explain_recommendation(user_id, mid)
        st.write(f"**{title}** (Predicted Score: {score:.2f})")
        st.caption(f"💡 {explanation}")
        