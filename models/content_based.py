import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def content_based_filtering():
    try:
        tag = pd.read_csv('data/tag.csv').sample(frac=0.5 , random_state=42)
        movies = pd.read_csv('data/movie.csv').sample(frac=0.5 , random_state=42)

    except FileNotFoundError:
        print("Error: CSV files not found.")
        return None
    
    movies['genres'] = movies['genres'].str.lower().str.replace('|', ' ', regex=False)
    df = pd.merge(movies , tag , on="movieId")
    df['tag'] = df['tag'].str.lower()

    df['tag'] = df['tag'].fillna('')

    movie_content = df.groupby(['movieId' , 'title' , 'genres'])['tag'].apply(lambda x: ' '.join(x)).reset_index()

    movie_content['metadata'] = movie_content['genres'] + ' ' + movie_content['tag']


    cv = CountVectorizer(stop_words='english')
    count_vec = cv.fit_transform(movie_content['metadata'])

    cosine_sim = cosine_similarity(count_vec , count_vec)

    return cosine_sim , movie_content

def get_recommendation(title , cosine_sim , movie_content):

    indices = pd.Series(movie_content.index , index=movie_content['title']).drop_duplicates()
    if title not in indices:
        return "Movie Not Found"

    # Get the index of the movie that matches the title
    idx = indices[title]
    

    # Get the pairwise similarity scores of all movies with that movie
    sim_score = list(enumerate(cosine_sim[idx]))

    sim_score = sorted(sim_score , key=lambda x: x[1] , reverse=True)

    sim_score = sim_score[1:7]

    movie_indices = [i[0] for i in sim_score]

    title = movie_content['title'].iloc[movie_indices]
    
    return title


# if __name__ =="__main__":

#     c , m = content_based_filtering()

#     title = get_recommendation("Redhead from Wyoming, The (1953)", c, m)

#     print(title)
