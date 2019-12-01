""" Character Type Recognition Model - 11/16/19"""
import sys
import pycrfsuite
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import operator
from decimal import ROUND_HALF_UP, Context
import spacy
import json
from hw3utils import *
from pymagnitude import Magnitude
from collections import defaultdict
from spacy.tokens import Doc, Token, Span
from typing import Iterable, Mapping, Sequence, Dict, Optional
from spacy.language import Language
import copy

def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:
    """ Creates a Spacy doc from json annotations; maps character indices to Spacy token indices."""
    entities = []
    labels = doc_json["labels"]
    doc = nlp(doc_json["text"])
    token_character_offsets = {tok.idx: tok.i for tok in doc}
    token_character_ends = {tok.idx + len(tok): tok.i for tok in doc}
    for label in labels:
        beg = label[0]
        end = label[1]
        ### cuts off trailing whitespace at start of a span ###
        if doc.text[beg]== " ":
            beg = beg + 1
        while beg not in token_character_offsets:
            beg -= 1
            if beg < 0:
                raise ValueError
        if end < len(doc.text):
            if doc.text[end-1] == " ":
                end = end - 1
            while end not in token_character_ends:
                end += 1
                if end > len(doc.text):
                    raise ValueError
        start = token_character_offsets[beg]
        end = token_character_ends[end]
        ent = Span(doc, start, end + 1, label=label[2])
        entities.append(ent)
    try:
        doc.ents = entities
    except ValueError:
        doc_ents = []
    return doc

def span_prf1_type_map(
    reference_docs: Sequence[Doc],
    test_docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:
    """Computes p/r/f/ for reference and test docs; optionally takes a "mapping" dict to collapse types."""
    prfs = {}
    values = defaultdict(lambda: defaultdict(int))
    zipped = zip(reference_docs, test_docs)
    for doc in zipped:
        if type_map is None:
            ref_ents = [(ent.label_, ent.start, ent.end) for ent in doc[0].ents]
            test_ents = [(ent.label_, ent.start, ent.end) for ent in doc[1].ents]
        else:
            ref_ents = [(type_map[ent.label_], ent.start, ent.end) if ent.label_ in type_map else (ent.label_, ent.start, ent.end) for ent in doc[0].ents]
            test_ents = [(type_map[ent.label_], ent.start, ent.end) if ent.label_ in type_map else (ent.label_, ent.start, ent.end) for ent in doc[1].ents]
        for ent in test_ents:
            if ent in ref_ents:
                values[ent[0]]['truepos'] += 1
            else:
                values[ent[0]]['falsepos'] += 1
        for ent in ref_ents:
            if ent not in test_ents:
                values[ent[0]]['falseneg'] += 1
    alltruepos = 0
    allfalsepos = 0
    allfalseneg = 0
    for ent in values:
        alltruepos += values[ent]['truepos']
        allfalsepos += values[ent]['falsepos']
        allfalseneg += values[ent]['falseneg']
        actual = values[ent]['truepos'] + values[ent]['falseneg']
        try:
            recall = values[ent]['truepos'] / actual
        except ZeroDivisionError:
            recall = 0.0
        predicted = values[ent]['truepos'] + values[ent]['falsepos']
        try:
            precision = values[ent]['truepos'] / predicted
        except ZeroDivisionError:
            precision = 0.0
        try:
            f1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f1 = 0.0
        prfs[ent] = PRF1(precision, recall, f1)
    try:
        precision_all = alltruepos / (alltruepos + allfalsepos)
    except ZeroDivisionError:
        precision_all = 0.0
    try:
        recall_all = alltruepos / (alltruepos + allfalseneg)
    except ZeroDivisionError:
        recall_all = 0.0
    try:
        f1_all = 2 * ((precision_all * recall_all) / (precision_all + recall_all))
    except ZeroDivisionError:
        f1_all = 0.0
    prfs[''] = PRF1(precision_all, recall_all, f1_all)
    return prfs

#---------------- Vector/Cluster Feature Extractors  ----------------#

class WordVectorFeature(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0) -> None:
        self.scaling = scaling
        self.vectors = Magnitude(vectors_path, normalized=False)
        self.keys = ['v'+repr(i) for i in range(self.vectors.dim)]

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
        idx: int,
    ) -> None:
        if relative_idx == 0:
            v = self.scaling * self.vectors.query(token)
            features.update(zip(self.keys,v))

class BrownClusterFeature(FeatureExtractor):
    def __init__(
        self,
        clusters_path: str,
        *,
        use_full_paths: bool = False,
        use_prefixes: bool = False,
        prefixes: Optional[Sequence[int]] = None,
    ):
        self.clusters_path = clusters_path
        self.clusters = defaultdict(int)
        if not use_full_paths and not use_prefixes:
            raise ValueError
        else:
            self.use_full_paths = use_full_paths
            self.use_prefixes = use_prefixes
            self.prefixes = prefixes
        with open(self.clusters_path) as f:
            for line in f:
                line = line.split()
                self.clusters[line[1]] = line[0]

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
        idx: int
    ) -> None:
        if relative_idx==0:
            path = self.clusters[token]
            pathstring = str(path)
            if self.use_full_paths:
                features[('cpath=%s' % path)] = 1.0
            if self.use_prefixes:
                if self.prefixes is None:
                    for i in range (1, len(pathstring) +1):
                        features["cprefix" + str(i) + "=" + pathstring[0:i]] = 1.0
                else:
                    for p in self.prefixes:
                        if p < len(pathstring):
                            features["cprefix" + str(p) + "=" + pathstring[0:p]] = 1.0


#---------------- Additional Feature Extractors ----------------#

class WindowedTokenFeatureExtractor:
    def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
        self.window_size = window_size
        self.feature_extractors = feature_extractors

    def extract(self, tokens: Sequence[str], idx: int) -> List[Dict[str, float]]:
        token_feature_dicts = []
        for i in range(len(tokens)):
            feature_dict = {}
            current_index = i
            for j in range(0, self.window_size + 1):
                for extractor in self.feature_extractors:
                    if len(tokens) - current_index > j:
                        focus_token = tokens[current_index +j]
                        extractor.extract(focus_token,current_index,j,tokens, feature_dict,idx)
                    if current_index - j >= 0:
                        focus_token = tokens[current_index - j]
                        extractor.extract(focus_token,current_index,-j, tokens, feature_dict,idx)
            token_feature_dicts.append(feature_dict)
        return token_feature_dicts

class BiasFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float],
            idx: int,
    ) -> None:
        if relative_idx == 0:
            features['bias'] = 1.0

class TokenFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
        idx: int,
    ) -> None:
        features['tok[' + repr(relative_idx) + ']=' + token] = 1.0

class CountFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
        idx: int,
    ) -> None:
        if relative_idx==0:
            if token.istitle():
                count = tokens.count(token)
                if count > 2:
                    features['count[' + repr(relative_idx) + '>2'] = 1.0


class SuffixFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float],
            idx: int,
    ) -> None:
        if relative_idx==0:
            suffixes = ('er','ar','ist','ian','ier','ur','eur','eer','ster', 'or','ess',)
            if token.endswith(suffixes) or token[:-1].endswith(suffixes):
                features['suffixfeature[' +repr(relative_idx) + ']'] = 1.0


class POSFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float],
            idx: int,
    ) -> None:
        if relative_idx == -1 or relative_idx==0:
            POS = nltk.pos_tag([token])
            features['pos[' + repr(relative_idx) +repr(POS[0][1])] = 1.0

class SentimentFeature(FeatureExtractor):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float],
            idx: int,
    ) -> None:
        scores = self.sid.polarity_scores(token)
        sentiment = max(scores.items(), key=operator.itemgetter(1))[0]
        if sentiment != 'compound' and sentiment !='neu':
            features['sentiment[' + repr(relative_idx) + sentiment] = 1.0

class PositionFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
        idx: int,
    ) -> None:
        if relative_idx==0:
            features['sent_position=' + repr(idx)] = 1.0

# class TitlecaseFeature(FeatureExtractor):
#     def extract(
#             self,
#             token: str,
#             current_idx: int,
#             relative_idx: int,
#             tokens: Sequence[str],
#             features: Dict[str, float],
#     ) -> None:
#         if relative_idx==0:
#             if token.istitle():
#                 features['titlecase[' + repr(relative_idx) + ']'] = 1.0


# class InitialTitlecaseFeature(FeatureExtractor):
#     def extract(
#             self,
#             token: str,
#             current_idx: int,
#             relative_idx: int,
#             tokens: Sequence[str],
#             features: Dict[str, float],
#     ) -> None:
#         if token.istitle() and ((current_idx + relative_idx)==0):
#             features['initialtitlecase[' + repr(relative_idx) + ']'] = 1.0

class WordShapeFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float],
            idx: int,
    ) -> None:
        UPPERCASE_RE = regex.compile(r"[\p{Lu}\p{Lt}]")
        LOWERCASE_RE = regex.compile(r"\p{Ll}")
        DIGIT_RE = regex.compile(r"\d")
        shape = regex.sub(UPPERCASE_RE,'X',token)
        shape = regex.sub(LOWERCASE_RE, 'x', shape)
        shape = regex.sub(DIGIT_RE, '0', shape)
        features['shape[' + repr(relative_idx) + ']=' + shape] = 1.0

#---------------- Encoder options and utils ----------------#

class BILOUEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> Sequence[str]:
        labels = [(token.ent_iob_ + '-' + token.ent_type_) if token.ent_iob_ else 'O' for token in tokens]
        length = len(labels)
        for i in range(0, length - 1):
            token = labels[i]
            entity = token[2:]
            label = token[:2]
            next = labels[i + 1]
            nextlabel = next[:2]
            nextentity = next[2:]
            if nextlabel == 'O' or nextentity != entity or nextlabel == 'B-':
                if label == 'B-':
                    labels[i] = 'U-' + entity
                if label == 'I-':
                    labels[i] = 'L-' + entity
        lasttoken = labels[length - 1]
        lastlabel = lasttoken[:2]
        lastentity = lasttoken[2:]
        if lastlabel == 'B-':
            labels[length - 1] = 'U-' + lastentity
        if lastlabel == 'I-':
            labels[length - 1] = 'L-' + lastentity
        return labels

class BIOEncoder(EntityEncoder):
    def encode(self,tokens: Sequence[Token]) -> Sequence[str]:
        return [(token.ent_iob_ + '-' + token.ent_type_) if token.ent_iob_ else 'O' for token in tokens]

class IOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> Sequence[str]:
        return [('I-' + token.ent_type_) if token.ent_iob_ else 'O' for token in tokens]

 #---------------- CRF ----------------#

class CRFsuiteEntityRecognizer:
    def __init__(
        self, feature_extractor: WindowedTokenFeatureExtractor, encoder: EntityEncoder
    ) -> None:
        self.feature_extractor = feature_extractor
        self.encoder_type = encoder
        self.tagger = None
        
    def encoder(self) -> EntityEncoder:
        return self.encoder_type

 #---------------- Decoding ----------------#

    def decode(self,labels: Sequence[str], tokens: Sequence[Token], doc: Doc) -> List[Span]:
        zipped = list(zip(tokens, labels))
        spans = []
        index1 = -1  # index1 keeps track of whether we've begun an entity at a particular index that we need to finish and add to the list. Set it to a dummy index if we aren't in the middle of an entity.
        i = 0
        current = zipped[i]
        while current[1] == 'O' and i < len(zipped) - 1:
            i += 1
            current = zipped[i]
        if i < len(zipped) - 1:
            last = current
            index1 = last[0].i
            i += 1
            current = zipped[i]
        while i < len(zipped) - 1:
            if index1 >= 0:
                current = zipped[i]
                current_index = current[0].i
                if extract_label(current) != 'O':
                    if extract_label(current) == 'B-' or extract_label(current) == 'U-' or extract_entity(
                            current) != extract_entity(last):
                        spans.append(Span(doc, index1, current_index, label=extract_entity(last)))
                        index1 = current_index
                        i += 1
                        last = current
                    if extract_label(current) == 'I-' or extract_label(current) == 'L-':
                        i += 1
                        last = current
                else:
                    spans.append(Span(doc, index1, current_index, label=extract_entity(last)))
                    index1 = -1
            else:
                while index1 == -1 and i < len(zipped) - 1:
                    current = zipped[i]
                    current_index = current[0].i
                    i += 1
                    if extract_label(current) != 'O':
                        index1 = current_index
                        last = current
        if index1 >= 0:
            current_index = current[0].i
            if current[1] == 'O':
                spans.append(Span(doc, index1, current_index, label=extract_entity(last)))
            else:
                spans.append(Span(doc, index1, current_index + 1, label=extract_entity(last)))
        return spans

    def train(self, docs: Iterable[Doc], algorithm: str, params: dict, path: str) -> None:
        trainer = pycrfsuite.Trainer(algorithm, verbose=False)
        trainer.set_params(params)
        encoder = self.encoder()
        for doc in docs:
            idx = 0
            for sent in doc.sents:
                tokens = list(sent)
                features = self.feature_extractor.extract([str(token) for token in tokens],idx)
                encoding = encoder.encode(tokens)
                trainer.append(features, encoding)
                idx +=1
        trainer.train(path)
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(path)

    def __call__(self, doc: Doc) -> Doc:
        entities = []
        if self.tagger is None:
            raise ValueError
        else:
            idx = 0
            for sent in doc.sents:
                tokens = [token for token in sent]
                features = self.feature_extractor.extract([str(token) for token in tokens],idx)
                predicted_labels = self.tagger.tag(features)
                ents = self.decode(predicted_labels, tokens, doc)
                entities.extend(ents)
                idx +=1
        doc.ents = entities
        return doc

    def predict_labels(self, tokens: Sequence[str]) -> List[str]:
        features = self.feature_extractor.extract(tokens)
        return self.tagger.tag(features)
        
def extract_label(token: tuple) -> str:
        return token[1][:2]

def extract_entity(token: tuple) -> str:
        return token[1][2:]


#---------------- Accuracy & Performance Analysis  ----------------#

def span_scoring_counts(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], type_map: dict = None
) -> ScoringCounts:
    truepos = []
    falsepos = []
    falseneg = []
    values = defaultdict(lambda: defaultdict(int))
    zipped = zip(reference_docs, test_docs)
    for doc in zipped:
        if type_map is None:
            ref_ents = [(ent.label_, ent.start, ent.end) for ent in doc[0].ents]
            test_ents = [(ent.label_, ent.start, ent.end) for ent in doc[1].ents]
        else:
            ref_ents = [(type_map[ent.label_], ent.start, ent.end) if ent.label_ in type_map else (ent.label_, ent.start, ent.end) for ent in doc[0].ents]
            test_ents = [(type_map[ent.label_], ent.start, ent.end) if ent.label_ in type_map else (ent.label_, ent.start, ent.end) for ent in doc[1].ents]
        for ent in doc[1].ents:
            label = ent.label_
            if type_map is not None and ent.label_ in type_map:
                label = type_map[ent.label_]
            rep = (label, ent.start, ent.end)
            if rep in ref_ents:
                values[label]['truepos'] += 1
                truepos.append(ScoringEntity(tuple(ent.text.split()),label))
            else:
                values[ent.label_]['falsepos'] += 1
                falsepos.append(ScoringEntity(tuple(ent.text.split()),ent.label_))
        for ent in doc[0].ents:
            if type_map is None:
                label = ent.label_
                if type_map and ent.label_ in type_map:
                    label = type_map[ent.label_]
            rep = (label, ent.start, ent.end)
            if rep not in test_ents:
                values[label]['falseneg'] += 1
                falseneg.append(ScoringEntity(tuple(ent.text.split()),label))
    return ScoringCounts(Counter(truepos),Counter(falsepos),Counter(falseneg))

#---------------- Data splits/Methods to annotate dev data and templatize it ----------------#

def predict_dev_labels(gold_dev):
    dev = copy.deepcopy(gold_dev)
    for doc in dev:
        doc.ents = []
    return [crf(doc) for doc in dev]

def compile_tagged_train_data(annotations):
    train = []
    for file in annotations:
        with open(file, encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                review = json.loads(line)
                doc = ingest_json_document(review, nlp)
                train.append(doc)
    return train[:560], train[560:]

def templatize(predicted_dev):
    templates = []
    for doc in predicted_dev:
        text = doc.text
        tokenized = nlp(text)
        template = list(text) ### all characters
        labels = []
        ents = doc.ents
        entity_map = defaultdict(lambda: defaultdict())
        for ent in ents:
            if ent.label_ not in entity_map.keys():
                entity_map[ent.label_][ent.text] = 0
            else:
                if ent.text not in entity_map[ent.label_].keys():
                    entity_map[ent.label_][ent.text] = max(entity_map[ent.label_].values()) + 1
            labels.append(ent.label_ + '_' + str(entity_map[ent.label_][ent.text]))
            start_token = tokenized[ent.start]
            end_token = tokenized[ent.end -1]
            start_token_character_offset = start_token.idx
            end_token_character_end = end_token.idx + len(end_token)
            for i in range(start_token_character_offset, end_token_character_end):
                template[i] = '_'
        templates.append((''.join(template),labels))
    return templates

def export_templates(dev_predicted):
    templates = templatize(dev_predicted)
    with open("Madlibs_Templates_linked.jsonl",'w') as f:
        for template in templates:
            f.write(json.dumps({"text": template[0], "labels": template[1]}) + "\n")

def export_entities(dev_predicted):
    entity_types = defaultdict(list)
    for doc in dev_predicted:
        for ent in doc.ents:
            entity_types[ent.label_].append(ent.text)
    with open("Madlibs_Entities.jsonl",'w') as f:
        f.write(json.dumps(entity_types))

def main():
    global nlp, crf, train, gold_dev
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    annotated_data = ['Annotation_task/Annotated_data/Jenny_annotations.jsonl',
                      'Annotation_task/Annotated_data/micaela_annotation.jsonl',
                      'Annotation_task/Annotated_data/molly_annotations.jsonl',
                      'Annotation_task/Annotated_data/qingwen_annotations.jsonl']
    crf = CRFsuiteEntityRecognizer(
        WindowedTokenFeatureExtractor(
            [
                WordVectorFeature("wiki-news-300d-1M-subword.magnitude", 1.0),
                TokenFeature(),
                SuffixFeature(),
                SentimentFeature(),
                WordShapeFeature(),
                # BiasFeature(),
                POSFeature(),
                CountFeature(),
                PositionFeature()
            ],
            1,
        ),
        BILOUEncoder(),
    )
    crf.tagger = pycrfsuite.Tagger()
    train, gold_dev = compile_tagged_train_data(annotated_data)
    crf.train(train, "ap", {"max_iterations": 100}, "tmp.model")
    dev_predicted = predict_dev_labels(gold_dev)
    scores = span_prf1_type_map(gold_dev, dev_predicted,type_map={"ANTAG":"FLAT_ANTAG", "FLAT": "FLAT_ANTAG"})
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    for ent_type, score in sorted(scores.items()):
        if ent_type == "":
            ent_type = "ALL"
        fields = [ent_type] + [
            str(rounder.create_decimal_from_float(num * 100)) for num in score
        ]
        print("\t".join(fields), file=sys.stderr)
    # Readable output for Span Scores ###
    span_scores = span_scoring_counts(gold_dev, dev_predicted,type_map ={"ANTAG":"FLAT_ANTAG", "FLAT": "FLAT_ANTAG"})
    print("False positives:")
    falsepos = span_scores[1]
    print(Counter([entity[1] for entity in falsepos]))
    print("")
    print("15 most common false positive entities:")
    print(falsepos.most_common(15))
    print("")
    falseneg = span_scores[2]
    print("15 most common false negative entities:")
    print(falsepos.most_common(15))
    occups = Counter([ent for ent in falseneg if ent[1]=='OCCUP'])
    locs = Counter([ent for ent in falseneg if ent[1] == 'LOC'])
    print("most common false negative occupations:")
    print(occups.most_common(15))
    print("most common false negative locs=")
    print(locs.most_common(15))

    export_templates(dev_predicted)
    export_entities(dev_predicted)
if __name__ == "__main__":
    main()