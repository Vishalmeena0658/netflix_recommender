import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from configs.config import Config
from src.data.loader import DatasetLoader
from src.data.eda import DataAnalyzer
from src.evaluation.metrics import Evaluator
from src.models.recommenders import PureSVDRecommender, ContentBasedRecommender, HybridRecommender

def main():
    df, movies_df = DatasetLoader.load_dataset()
    DataAnalyzer.run_mandatory_eda(df, movies_df)
    
    train_df, test_df = train_test_split(df, test_size=Config.TEST_SIZE, random_state=Config.SEED)
    
    print("[Pipeline] Fitting SVD, Content-Based, and Hybrid Models...")
    svd, cb = PureSVDRecommender(), ContentBasedRecommender()
    svd.fit(train_df); cb.fit(train_df, movies_df)
    hybrid = HybridRecommender(svd, cb, movies_df)
    
    eval_sample = test_df.sample(n=min(2000, len(test_df)), random_state=Config.SEED)
    y_true = eval_sample['rating'].values
    
    leaderboard = []
    for name, model in [("Pure-SVD", svd), ("Content-Based", cb), ("Hybrid Ensemble", hybrid)]:
        preds = [model.predict(r['user_id'], r['movie_id']) for _, r in eval_sample.iterrows()]
        err = Evaluator.evaluate_predictions(y_true, preds)
        
        user_maps, user_ndcgs = [], []
        for uid in eval_sample['user_id'].unique()[:10]:
            # Movie is relevant if actual user rating >= 3.5 
            actual_liked = set(test_df[(test_df['user_id'] == uid) & (test_df['rating'] >= 3.5)]['movie_id'])
            if not actual_liked: continue
            
            movie_scores = [(mid, model.predict(int(uid), int(mid))) for mid in train_df['movie_id'].unique()[:200]]
            top_10 = [mid for mid, _ in sorted(movie_scores, key=lambda x: x[1], reverse=True)[:10]]
            
            user_maps.append(Evaluator.calculate_map_at_k(actual_liked, top_10, k=10))
            user_ndcgs.append(Evaluator.calculate_ndcg_at_k(actual_liked, top_10, k=10))
            
        leaderboard.append([name, f"{err['RMSE']:.4f}", f"{err['MAE']:.4f}", f"{np.mean(user_maps):.4f}", f"{np.mean(user_ndcgs):.4f}"])
        
    print("\n🏆 MANDATORY MODEL COMPARISON")
    print(tabulate(leaderboard, headers=["Model", "RMSE", "MAE", "MAP@10", "NDCG@10"], tablefmt="fancy_grid"))

    print("\n🔍 MANDATORY QUALITY ASSURANCE (Success & Failure)")
    eval_sample['pred'] = [hybrid.predict(r['user_id'], r['movie_id']) for _, r in eval_sample.iterrows()]
    eval_sample['err'] = np.abs(eval_sample['rating'] - eval_sample['pred'])
    eval_sample = eval_sample.merge(movies_df, on='movie_id', how='left')
    
    print("\n✅ Top SUCCESS CASE:")
    success = eval_sample.nsmallest(1, 'err').iloc[0]
    print(f"User {success['user_id']} -> '{success['title']}' | Actual: {success['rating']} | Pred: {success['pred']:.2f}")
    print(f"💡 Explanation: {hybrid.explain_recommendation(success['user_id'], success['movie_id'])}")

    print("\n❌ Top FAILURE CASE:")
    failure = eval_sample.nlargest(1, 'err').iloc[0]
    print(f"User {failure['user_id']} -> '{failure['title']}' | Actual: {failure['rating']} | Pred: {failure['pred']:.2f}")

if __name__ == "__main__":
    main()