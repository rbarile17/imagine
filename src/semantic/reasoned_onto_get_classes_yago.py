import os

import pandas as pd

from ast import literal_eval

from . import DB50K_PATH, DB50K_REASONED_PATH
from . import YAGO4_20_PATH, YAGO4_20_REASONED_PATH

from ..dataset import Dataset

dataset = Dataset(dataset="YAGO4-20")

entities = set(dataset.entity_to_id.keys())

uris = set([f"http://yago-knowledge.org/resource/{entity}"for entity in entities])

uris_string = " ".join([f"<{uri}>" for uri in uris])

query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?entity  ?class
    WHERE {{
        VALUES ?entity {{{uris_string}}} .
        OPTIONAL {{
            ?entity rdf:type ?class .
            FILTER(?class != <http://www.w3.org/2002/07/owl#NamedIndividual>)
            FILTER(?class != <http://www.w3.org/2002/07/owl#Thing>)
            FILTER(?class != <http://schema.org/Thing>)
        }}
    }}
"""

with open("query.sparql", "w") as f:
    f.write(query)

robot_command = f"""
sudo robot query --input {str(YAGO4_20_PATH / "YAGO4-20.nt")} --query query.sparql results.csv
"""

os.system(robot_command)

# read results.csv using empty set as placeholder for nan in the class column
results = pd.read_csv("results.csv", na_filter=False)

results = results.groupby("entity", as_index=False)["class"].agg(set)
results = results.rename(columns={"class": "classes"})
results["entity"] = results["entity"].map(
    lambda entity: entity.split("http://yago-knowledge.org/resource/")[-1]
)

results.to_csv(YAGO4_20_REASONED_PATH / "entities.csv", index=False)

os.remove("query.sparql")
os.remove("results.csv")
