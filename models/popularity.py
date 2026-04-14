import pandas as pd
import numpy as np


def load_data():

    try:
        movies = pd.read_csv("data/movie.csv").sample(frac=0.5 , random_state=42)
        ratings = pd.read_csv("data/rating.csv").sample(frac=0.5 , random_state=42)

    except FileNotFoundError:
        print("Error: CSV files not found.")

    # movies_data = pd.merge(ratings , movies , on="movieId")

    stats = ratings.groupby('movieId').agg(
        avg_rating=('rating' , 'mean'),
        rating_count=('rating' , 'count')
    ).reset_index()

    popular_stats = stats[(stats['avg_rating'] > 4) & (stats['rating_count'] > 15000)].copy()
    popular_stats['avg_rating'] = popular_stats['avg_rating'].round(1)

    # Merge only the high-quality matches
    # 'inner' join here is faster if you only want the popular ones
    result = movies.merge(popular_stats, on='movieId', how='inner')
    result['year'] = result['title'].str.extract(r'\((\d{4})\)')
    result['clean_title'] = result['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    results = result.sort_values(by='avg_rating', ascending=False)

    return results

# if __name__ =="__main__":

#     data = load_data()

#     print(data)
