import time

import pandas as pd

from tqdm import tqdm
from urllib.parse import unquote, quote

from . import sparql

from ..dataset import Dataset


dbpedia_url_base = "http://dbpedia.org/resource/"
get_entity = lambda row: row["entity_uri"]["value"].replace(dbpedia_url_base, "")
get_class = lambda row: row["class"]["value"].split("/")[-1]


def to_set(classes):
    classes = set(classes.tolist())
    classes = classes if classes != {""} else set()

    return classes


def custom_unquote(url):
    decoded_url = unquote(url)
    reencoded_url = decoded_url.replace('"', "%22")
    return reencoded_url


def custome_quote(url):
    url = url.replace("%22", '"')
    return quote(url, safe='()!/,:*$')


def get_classes(entities):
    entities = [custom_unquote(entity) for entity in entities]
    entities = [f"<http://dbpedia.org/resource/{entity}>" for entity in entities]
    values = f"{{{' '.join(entities)}}}"

    query = f"""
        SELECT DISTINCT ?entity_uri (COALESCE(?class, "") as ?class)
        WHERE {{
            VALUES ?entity_uri {values}

            OPTIONAL {{
                ?entity_uri a ?class .
                FILTER strstarts(str(?class), "http://dbpedia.org/ontology/")
                FILTER NOT EXISTS {{
                    ?subClass rdfs:subClassOf ?class .
                    ?entity_uri a ?subClass .
                    FILTER (?subClass != ?class)
                }}
            }}
        }}
    """
    sparql.setQuery(query)

    rows = sparql.queryAndConvert()["results"]["bindings"]

    return [
        {"class": get_class(row), "entity": custome_quote(get_entity(row))}
        for row in rows
    ]


dataset = Dataset(dataset="DB100K")

entities = list(dataset.entity_to_id.keys())
literals = [
    "?autoplay=true",
    "en",
    "index.html",
    "index.php",
    "listen.pls",
    "player.html",
    "unk",
    "www.monolithgraphics.com",
]

for literal in literals:
    entities.remove(literal)

batch_size = 80
num_entities = len(entities)
batch_start = 0
classes = []
while batch_start < num_entities:
    batch_end = min(num_entities, batch_start + batch_size)
    cur_batch = entities[batch_start:batch_end]

    classes.extend(get_classes(cur_batch))

    batch_start += batch_size
    time.sleep(1)

    tqdm.write(f"Processed {batch_end} entities out of {num_entities}")


classes = pd.DataFrame(classes)
classes = classes.groupby(by=classes["entity"]).agg(classes=("class", to_set))
classes = classes.reset_index()

classes.to_csv("entities.csv", index=False, header=True)
