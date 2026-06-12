import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from configs.config import Config

class PureSVDRecommender:
    def __init__(self, n_factors=Config.SVD_FACTORS):
        self.n_factors = n_factors
        
    def fit(self, train_df: pd.DataFrame):
        self.global_mean = train_df['rating'].mean()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(train_df['user_id'].unique())}
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(train_df['movie_id'].unique())}
        
        R = csr_matrix((train_df['rating'].values - self.global_mean, 
                       (train_df['user_id'].map(self.user_to_idx).values, 
                        train_df['movie_id'].map(self.movie_to_idx).values)), 
                       shape=(len(self.user_to_idx), len(self.movie_to_idx)))
        
        U, s, Vt = svds(R, k=min(self.n_factors, R.shape[1] - 1))
        self.user_vecs, self.item_vecs = U @ np.diag(np.sqrt(s)), np.diag(np.sqrt(s)) @ Vt

    def predict(self, user_id, movie_id):
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx: return self.global_mean
        return np.clip(self.global_mean + np.dot(self.user_vecs[self.user_to_idx[user_id], :], self.item_vecs[:, self.movie_to_idx[movie_id]]), 1.0, 5.0)

class ContentBasedRecommender:
    def fit(self, train_df, movies_df):
        self.global_mean = train_df['rating'].mean()
        movies_df['title_clean'] = movies_df['title'].fillna('')
        self.sim_matrix = cosine_similarity(TfidfVectorizer(stop_words='english').fit_transform(movies_df['title_clean']), dense_output=False)
        self.movie_to_idx = {row['movie_id']: idx for idx, row in movies_df.iterrows()}
        self.user_history = {uid: dict(zip(g['movie_id'], g['rating'])) for uid, g in train_df.groupby('user_id')}

    def predict(self, user_id, movie_id):
        if movie_id not in self.movie_to_idx or user_id not in self.user_history: return self.global_mean
        sim_scores = self.sim_matrix[self.movie_to_idx[movie_id]].toarray().flatten()
        weights, scores = [], []
        for h_mid, rating in self.user_history[user_id].items():
            if h_mid in self.movie_to_idx:
                sim = sim_scores[self.movie_to_idx[h_mid]]
                if sim > 0: weights.append(sim); scores.append(sim * rating)
        return np.clip(sum(scores) / sum(weights), 1.0, 5.0) if weights else self.global_mean

class HybridRecommender:
    def __init__(self, svd, cb, movies_df, alpha=Config.HYBRID_ALPHA):
        self.svd, self.cb, self.movies_df, self.alpha = svd, cb, movies_df, alpha

    def predict(self, user_id, movie_id):
        # Cold Start Fallback Check
        if user_id not in self.svd.user_to_idx:
            return self.cb.predict(user_id, movie_id) # Rely entirely on Content if user is new
        return (self.alpha * self.svd.predict(user_id, movie_id)) + ((1 - self.alpha) * self.cb.predict(user_id, movie_id))
    
    def explain_recommendation(self, user_id, target_movie_id):
        if user_id not in self.cb.user_history:
            return "Recommended based on general popularity (Cold Start logic applied)."
        user_ratings = self.cb.user_history[user_id]
        best_past_movie = max(user_ratings, key=user_ratings.get)
        
        t_title = self.movies_df[self.movies_df['movie_id'] == target_movie_id]['title'].values[0]
        p_title = self.movies_df[self.movies_df['movie_id'] == best_past_movie]['title'].values[0]
        return f"Because you enjoyed '{p_title}', you are likely to enjoy '{t_title}'."