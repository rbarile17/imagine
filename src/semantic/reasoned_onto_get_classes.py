import os

import pandas as pd

from ast import literal_eval

from . import DB100K_PATH, DB100K_REASONED_PATH

entities = pd.read_csv(
    DB100K_PATH / "entities.csv",
    converters={"classes": literal_eval}
)

uris = set(
    entities["entity"]
    .map(lambda entity: f"http://dbpedia.org/resource/{entity}")
    .values
)

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
    }}
}}
"""

with open("query.sparql", "w") as f:
    f.write(query)

robot_command = f"""
sudo robot query --input {str(DB100K_REASONED_PATH / "DBpedia50_reasoned.owl")} --query query.sparql results.csv
"""

os.system(robot_command)

# read results.csv using empty set as placeholder for nan in the class column
results = pd.read_csv("results.csv", na_filter=False)

results = results.groupby("entity", as_index=False)["class"].agg(set)
results = results.rename(columns={"class": "classes"})
results["entity"] = results["entity"].map(
    lambda entity: entity.split("http://dbpedia.org/resource/")[-1]
)

results.to_csv(DB100K_REASONED_PATH / "entities.csv", index=False)

os.remove("query.sparql")
os.remove("results.csv")
