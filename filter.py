import os
import csv

full_path = os.path.join('.', 'movie_data.csv')
filtered_path = os.path.join('.', 'filtered_movies.csv')

with open(full_path, 'r') as input_path:
    reader = csv.DictReader(input_path)
    with open(filtered_path, 'w') as filtered:
        writer = csv.DictWriter(filtered, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            genres = [x.lower() if not x.endswith(',') else x[:-1].lower() for x in row['genre'].split()]
            plot = row['plot'].split()
            if not ('adult' in genres or 'biography' in genres or 'documentary' in genres) and len(plot) > 30:
                writer.writerow(row)