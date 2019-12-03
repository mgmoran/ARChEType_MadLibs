import os
from typing import List
import json
import re
import werkzeug
import html


class MadLib:
    def __init__(self, title: str, genres: List[str], plot: str, blanks: List[str]):
        self.title = title
        self.genres = genres
        self.plot = plot
        self.blanks = blanks


class Database:
    def __init__(self, template_path: str, entities_path: str):
        self.all_madlibs = {}
        self.entities = {}
        self.template_path = template_path
        self.entities_path = entities_path
        self.set_up()

    def set_up(self):
        with open(self.template_path, 'r') as templates:
            f = templates.readlines()
            for line in f:
                json_read = json.loads(line)
                if not json_read["labels"]:
                    continue
                movie_id, title, g, plot = json_read['text'].split('\t')
                title = title[7:]
                g = g[7:]
                genres = g.split(' ')
                plot = plot[6:]
                madlib = MadLib(title, genres, plot, json_read["labels"])
                self.all_madlibs[title] = madlib

        with open(self.entities_path, 'r') as ents:
            f = ents.read()
            self.entities = json.loads(f)

    def add_madlib(self, ml: MadLib):
        self.all_madlibs[ml.title] = ml


def fill_madlib(plot: str, blanks: List[str]):
    b = [f"<b>{bl}</b>" for bl in blanks]
    to_fill = re.sub('__*', '%s', plot)
    filled_in = to_fill % tuple(b)
    formatted = filled_in
    return formatted
