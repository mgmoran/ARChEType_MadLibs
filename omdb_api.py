import omdb

# omdb is a pythom wrapper for OMDB written by a third party
# run pip install omdb if omdb is not already in your python environment

# api key is user specifc. Micaela has one (limit 1000 a day)
key = "60f61a52"
omdb.set_default('apikey', key)


res = omdb.title('Avengers', fullplot=True)
title = res['title']
plot = res['plot']
print(f'movie: {title}')
print(plot)