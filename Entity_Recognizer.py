""" Character Type Recognition Model - 11/16/19"""
import spacy
import json
from spacy.tokens import Doc, Token, Span
from typing import Iterable, Mapping, Sequence, Dict, Optional
from spacy.language import Language


def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:
    entities = []
    labels = doc_json["labels"]
    doc = nlp(doc_json["text"])
    vocab = [word.text for word in doc.vocab]
    token_character_offsets = {tok.idx: tok.i for tok in doc}
    token_character_ends = {tok.idx + len(tok): tok.i for tok in doc}
    for label in labels:
        beg = label[0]
        end = label[1]
        while beg not in token_character_offsets:
            beg -= 1
            if beg < 0:
                raise ValueError
        if end < len(doc.text):
            if doc.text[end - 1] != " ":
                while end not in token_character_ends:
                    end += 1
                    if end > len(doc.text):
                        raise ValueError
            else:
                end -= 1
        start = token_character_offsets[beg]
        end = token_character_ends[end]
        ent = Span(doc, start, end + 1, label=label[2])
        entities.append(ent)
    doc.ents = entities
    return doc

def compile_tagged_train_data(annotations):
    train = []
    for file in annotations:
        with open(file, encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                review = json.loads(line)
                doc = ingest_json_document(review, nlp)
                train.append(doc)
    return train



if __name__=='__main__':
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    annotated_data = ['Jenny_annotations.jsonl', 'micaela_annotation.jsonl', 'molly_annotations.jsonl', 'qingwen_annotations.jsonl']
    train = compile_tagged_train_data(annotated_data)
