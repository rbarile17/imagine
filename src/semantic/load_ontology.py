import pandas as pd

from tqdm import tqdm
from ast import literal_eval

from owlready2 import *

from .. import DB50K_PATH, DB100K_PATH
from ..dataset import Dataset

tqdm.pandas()


def check_disjoint_classes(classes):
    if len(classes) < 2:
        return False
    onto_classes = []
    for class_ in classes:
        try:
            onto_classes.append(onto[class_])
        except TypeError:
            pass
    onto_classes = [class_ for class_ in onto_classes if class_ is not None]
    classes_and_ancestors = []
    for class_ in onto_classes:
        classes_and_ancestors.append(class_)
        classes_and_ancestors.extend(class_.ancestors())
    for class_ in onto_classes:
        for disjoint_class in class_.disjoints():
            other_class = (
                disjoint_class.entities[0]
                if disjoint_class.entities[0] != class_
                else disjoint_class.entities[1]
            )

            if other_class in classes_and_ancestors:
                return True

        for ancestor in class_.ancestors():
            if ancestor.name != "Thing":
                for disjoint_class in ancestor.disjoints():
                    other_class = (
                        disjoint_class.entities[0]
                        if disjoint_class.entities[0] != ancestor
                        else disjoint_class.entities[1]
                    )
                    if other_class in classes_and_ancestors:
                        return True

    return False


onto_path.append(str(DB100K_PATH))
onto = get_ontology("DBpedia.owl")

onto = onto.load()

dbr = onto.get_namespace("http://dbpedia.org/resource/")

converters = {"classes": literal_eval}
entities = pd.read_csv(DB100K_PATH / "entities.csv", converters=converters)

entities["disjoint"] = entities["classes"].progress_map(check_disjoint_classes)
entities.loc[entities["disjoint"], "classes"] = None
entities = entities.drop(columns=["disjoint"])
entity_names = entities["entity"].tolist()
for _, (entity, classes) in entities.iterrows():
    instantiated_classes = 0
    classes = set() if classes is None else classes
    for class_ in classes:
        try:
            with dbr:
                onto[class_](entity)
            instantiated_classes += 1
        except TypeError:
            # skip if class is not in ontology
            pass

    if instantiated_classes == 0:
        with dbr:
            Thing(entity)

dataset = Dataset(dataset="DB100K")

for triple in tqdm(dataset.training_triples):
    s, p, o = dataset.labels_triple(triple)
    if s in entity_names and o in entity_names:
        with dbr:
            individual = dbr[s]
            if hasattr(individual, p):
                getattr(individual, p).append(dbr[o])

onto.save(str((DB100K_PATH / "DB100K.owl")))
