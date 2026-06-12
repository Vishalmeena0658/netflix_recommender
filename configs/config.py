import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE_DIR, 'data', 'raw')

class Config:
    DATA_DIR = DATA_DIR
    DATA_FILES = [os.path.join(DATA_DIR, f'combined_data_{i}.txt') for i in range(1, 5)]
    MOVIES_FILE = os.path.join(DATA_DIR, 'movie_titles.csv')
    
    ONLY_FIRST_FILE = True
    SAMPLE_SIZE = 1_500_000 
    SEED = 42
    TEST_SIZE = 0.2
    
    SVD_FACTORS = 15
    KNN_NEIGHBORS = 30
    HYBRID_ALPHA = 0.7