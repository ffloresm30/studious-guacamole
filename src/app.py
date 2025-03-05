from utils import db_connect
engine = db_connect()

# your code here

import pandas as pd
import ast
from utils import db_connect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Cargando datos 
url_movies = "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv"
url_credits = "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv"


movies_df = pd.read_csv(url_movies)
credits_df = pd.read_csv(url_credits)

# Verificando nombres de columnas
print(movies_df.columns)
print(credits_df.columns)

# Creando bases de datos 
engine = db_connect()


movies_df.to_sql('movies', engine, if_exists='replace', index=False)
credits_df.to_sql('credits', engine, if_exists='replace', index=False)


query = """
SELECT
    m.id AS movie_id,
    m.title,
    m.overview,
    m.genres,
    m.keywords,
    c.cast,
    c.crew
FROM
    movies m
JOIN
    credits c ON m.title = c.title
"""


combined_df = pd.read_sql(query, engine)


def process_column(data, column_name):
    return data.apply(lambda x: [item['name'] for item in ast.literal_eval(x)] if isinstance(x, str) else [])

combined_df['genres'] = process_column(combined_df['genres'], 'genres')
combined_df['keywords'] = process_column(combined_df['keywords'], 'keywords')
combined_df['cast'] = combined_df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]] if isinstance(x, str) else [])
combined_df['crew'] = combined_df['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if isinstance(x, str) else [])
combined_df['overview'] = combined_df['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

for column in ['genres', 'cast', 'crew', 'keywords']:
    combined_df[column] = combined_df[column].apply(lambda x: [i.replace(" ", "") for i in x])

combined_df['tags'] = combined_df.apply(lambda row: ' '.join(row['overview']) + ' ' + ' '.join(row['genres']) + ' ' + ' '.join(row['keywords']) + ' ' + ' '.join(row['cast']) + ' ' + ' '.join(row['crew']), axis=1)

print(combined_df.head())



# Vectorizandp  texto de la columna 'tags'
cv = CountVectorizer(max_features=5000, stop_words='english')
X = cv.fit_transform(combined_df['tags']).toarray()


# Crear etiquetas para el modelo KNN
y = combined_df['movie_id']

# Crear el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(X, y)

#Funcion para obtener las peliculas más recomendadas

def recommend(movie_title):
    movie_index = combined_df[combined_df['title'] == movie_title].index[0]
    movie_vector = X[movie_index].reshape(1, -1)
    distances, indices = knn.kneighbors(movie_vector)
    recommended_movie_indices = indices.flatten()[1:6]
    
    for i in recommended_movie_indices:
        print(combined_df.iloc[i]['title'])


recommend("The Dark Knight Rises")
