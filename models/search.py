import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv('data/movie.csv').sample(frac=0.5 , random_state=42)      # movieId, title, genres
ratings = pd.read_csv('data/rating.csv').sample(frac=0.5 , random_state=42)     # userId, movieId, rating

# Compute popularity stats per movie
stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    rating_count=('rating', 'count')
).reset_index()

movies = movies.merge(stats, on='movieId', how='left')
movies['avg_rating'] = movies['avg_rating'].fillna(0)
movies['rating_count'] = movies['rating_count'].fillna(0)

# Extract year from title e.g. "Toy Story (1995)"
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)

movies['search_text'] = movies['clean_title'] + ' ' + movies['genres'].str.replace('|', ' ')

vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='word')
tfidf_matrix = vectorizer.fit_transform(movies['search_text'])

def search(query, top_n=10, genre_filter=None):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    movies['score'] = scores
    results = movies[movies['score'] > 0.3].copy()

    # Optional genre filter
    if genre_filter:
        results = results[results['genres'].str.contains(genre_filter, case=False)]

    # Rank by TF-IDF score + boost popular/highly rated movies
    results['final_score'] = (
        results['score'] * 0.6 +
        (results['avg_rating'] / 5) * 0.25 +
        (np.log1p(results['rating_count']) / 10) * 0.15
    )

    return results.sort_values('final_score', ascending=False)[
        ['title', 'genres', 'avg_rating', 'rating_count', 'final_score']
    ].head(top_n)

# Trie-based prefix autocomplete
class TrieNode:
    def __init__(self):
        self.children = {}
        self.movies = []  # store (popularity_score, title, movieId)

class AutoComplete:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, title, movie_id, popularity):
        node = self.root
        for char in title.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.movies.append((popularity, title, movie_id))

    def search(self, prefix, top_n=5):
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        # Sort by popularity descending
        sorted_movies = sorted(node.movies, key=lambda x: -x[0])
        seen, results = set(), []
        for _, title, mid in sorted_movies:
            if mid not in seen:
                seen.add(mid)
                results.append({'movieId': mid, 'title': title})
            if len(results) == top_n:
                break
        return results
    
ac = AutoComplete()
for _, row in movies.iterrows():
    popularity = row['avg_rating'] * np.log1p(row['rating_count'])
    ac.insert(row['clean_title'], row['movieId'], popularity)

if __name__ =="__main__":

 
    print("\n=== Autocomplete for 'no' ===")
    print(ac.search('no'))