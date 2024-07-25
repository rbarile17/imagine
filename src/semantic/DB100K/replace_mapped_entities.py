import pandas as pd

from ... import DB100K_PATH

from ...dataset import Dataset

dataset = Dataset(dataset="DB100K")
mapping = pd.read_csv(DB100K_PATH / "mapping.csv")
mapping = mapping.set_index("entity")["dbpedia"].to_dict()


def map_triples(triples):
    mapped_triples = []
    for triple in triples:
        s, p, o = dataset.labels_triple(triple)
        if s in mapping and (o in mapping or "Q" not in o):
            s = mapping[s]
            o = mapping[o] if "Q" in o else o
            mapped_triples.append({"s": s, "p": p, "o": o})

    mapped_triples = pd.DataFrame(mapped_triples)
    return mapped_triples


mapped_triples = map_triples(dataset.training_triples)
mapped_triples.to_csv("train.txt", sep="\t", index=False, header=False)

mapped_triples = map_triples(dataset.validation_triples)
mapped_triples.to_csv("valid.txt", sep="\t", index=False, header=False)

mapped_triples = map_triples(dataset.testing_triples)
mapped_triples.to_csv("test.txt", sep="\t", index=False, header=False)
