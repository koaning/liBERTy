import time 
import numpy as np
from datasets import load_dataset
from memo import memfile, grid

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import HashingVectorizer

import datasets

datasets.logging.set_verbosity_error()

if __name__ == "__main__":

    @memfile("sgd-benchmarks.jsonl", skip=True)
    def train_test_sgd_model(dataset_name, n_features=10_000, n_hash=3, subword=True, early_stopping=False):
        datasets = {
            "ag_news": {"train": "train", "valid": "test", "text": "text", "label": "label"},
            "banking77": {"train": "train", "valid": "test", "text": "text", "label": "label"},
            "emotion": {"train": "train", "valid": "test", "text": "text", "label": "label"},
        }
        dataset = load_dataset(dataset_name)
        d = datasets[dataset_name]

        dataset = load_dataset(dataset_name)
        X_train = dataset[d['train']][d['text']]
        y_train = dataset[d['train']][d['label']]
        X_test  = dataset[d['valid']][d['text']]
        y_test  = dataset[d['valid']][d['label']]

        featurizers = [HashingVectorizer(n_features=n_features + i) for i in range(n_hash)]

        if subword:
            featurizers += [HashingVectorizer(ngram_range = (2, 4), n_features=n_features + i, analyzer="char") for i in range(n_hash)]

        classifier = SGDClassifier()
        if early_stopping:
            classifier = SGDClassifier(early_stopping=True, n_iter_no_change=3, tol=0.0001, validation_fraction=0.2)
        
        pipe = make_pipeline(
            make_union(*featurizers),
            classifier
        )

        t0 = time.time()
        pipe.fit(X_train, y_train)
        t1 = time.time()
        pred_test = pipe.predict(X_test)
        t2 = time.time()
        pred_train = pipe.predict(X_train)

        return {
            "acc_valid": np.mean(pred_test == y_test),
            "acc_train": np.mean(pred_train == y_train),
            "f1_valid": f1_score(pred_test, y_test, average="weighted"),
            "f1_train": f1_score(pred_train, y_train, average="weighted"),
            "train_time": t1 - t0,
            "pred_time": t2 - t1, 
            "n_items": len(X_train)
        }

    settings = grid(
        dataset_name=["emotion", "banking77", "ag_news"], 
        n_features=[1000, 2000, 5000, 10_000, 20_000], 
        n_hash=[1, 2, 3, 4, 5],
        subword=[True, False],
        early_stopping=[True, False]
    )

    for settings in settings:
        result = train_test_sgd_model(**settings)
        if not result:
            print("skipped!")
