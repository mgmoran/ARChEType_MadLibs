import os
from typing import List
import json


class MadLib:
    def __init__(self, title: str, genres: List[str], plot: str, blanks: List[str]):
        self.title = title
        self.genres = genres
        self.plot = plot
        self.blanks = blanks


class Database:
    def __init__(self, template_path: str, entities_path: str):
        self.all_madlibs = {}
        self.template_path = template_path
        self.entities_path = entities_path
        # self.set_up()

    def set_up(self):
        with open(self.template_path, 'r') as templates:
            f = templates.readlines()
            for line in f:
                json_dict = json.loads(line)
                madlib = MadLib(json_dict["title"], json_dict["genre"], json_dict["plot"], json_dict["blanks"])
                self.all_madlibs[json_dict["title"]] = madlib

    def add_madlib(self, ml: MadLib):
        self.all_madlibs[ml.title] = ml
