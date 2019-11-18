import omdb
import os
import csv
import pandas as pd

# omdb is a pythom wrapper for OMDB written by a third party
# run pip install omdb if omdb is not already in your python environment

# Define movie lists
basics_path = os.path.join('.', 'title.akas.tsv')
ratings_path = os.path.join('.', 'title.ratings.tsv')

basics_df = pd.read_csv(basics_path, sep='\t')
eq = basics_df.iloc[3]['region']
basics_df.drop(basics_df[basics_df['region'] != eq].index, inplace=True)
col_to_drop = [x for x in basics_df.columns if x not in ['titleId', 'title']]
basics_df = basics_df.drop(columns=col_to_drop, axis=1)

ratings_df = pd.read_csv(ratings_path, sep='\t')
indexNames = ratings_df[(ratings_df['averageRating'] < 9.0) & (ratings_df['numVotes'] <= 500)].index
ratings_df.drop(indexNames, inplace=True)
usable_df = pd.merge(basics_df, ratings_df, on=['titleId'])


og_csv = os.path.join('.', 'movie_data.csv')
output_path = os.path.join('.', 'rom_act.csv')
movies = usable_df['title'].tolist()

movies_to_try = movies[948:]
# api key is user specifc. Micaela has one (limit 1000 a day)
key = "60f61a52"
omdb.set_default('apikey', key)


with open(og_csv) as already_done:
    dr = csv.DictReader(already_done)
    already_have = [x['title'] for x in dr]

already_have = set(already_have)
movies_to_try = [x for x in movies_to_try if not x in already_have]

counter = 1
for m in movies_to_try:
    print(f'movie: {m}')
    movie_dict = {}
    try:
        res = omdb.title(m, fullplot=True)
        movie_dict['title'] = res['title']
        movie_dict['genre'] = res['genre']
        movie_dict['plot'] = res['plot']
        df = pd.DataFrame([movie_dict], columns=movie_dict.keys())
        output_path = os.path.join('.', 'movie_data.csv')
        genres = [x.lower() if not x.endswith(',') else x[:-1].lower() for x in movie_dict['genre'].split()]
        plot = movie_dict['plot'].split()

        #OG FILTER
        if ('adult' in genres or 'biography' in genres or 'documentary' in genres) and len(plot) <= 30:
            continue

        # 11/17 filter -- action and romance only
        if (not 'action' in genres) and (not 'romance' in genres):
            continue

        df = pd.DataFrame([movie_dict], columns=movie_dict.keys())
        with open(output_path, 'a', newline='') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=["title", "genre", "plot"])
            if counter == 1:
                writer.writeheader()
            writer.writerow(movie_dict)
        print(f"added movie {counter}")
        counter += 1
    except KeyError as e:
        print(f'response = {res}')

