import time

import pandas as pd

from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

from ...dataset import Dataset

from ... import DB100K_PATH


sparql = SPARQLWrapper(endpoint="https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

get_entity = lambda row: row["entity_uri"]["value"].split("/")[-1]
dbpedia_url_base = "http://dbpedia.org/resource/"
get_dbpedia = lambda row: row["dbpedia"]["value"].replace(dbpedia_url_base, "")


def to_set(values):
    values = set(values.tolist())
    values = values if values != {""} else set()

    return values

def get_dbpedia_same_as(entities):
    entities = [f"wd:{entity}" for entity in entities]
    values = f"{{{' '.join(entities)}}}"

    query = f"""
        SELECT DISTINCT ?entity_uri ?dbpedia
        WHERE {{
            VALUES ?entity_uri {values} .
            ?sitelink schema:about ?entity_uri ;
                schema:isPartOf <https://en.wikipedia.org/> .
            BIND(URI(CONCAT("{dbpedia_url_base}", SUBSTR(STR(?sitelink), 31))) as ?dbpedia)
        }}
    """

    sparql.setQuery(query)

    rows = sparql.queryAndConvert()["results"]["bindings"]
    return [{"entity": get_entity(row), "dbpedia": get_dbpedia(row)} for row in rows]


dataset = Dataset(dataset="DB100K")
entities = list(dataset.entity_to_id.keys())

entities = [e for e in entities if 'Q' in e]

batch_size = 250
num_entities = len(entities)
batch_start = 0
results = []
while batch_start < num_entities:
    batch_end = min(num_entities, batch_start + batch_size)
    cur_batch = entities[batch_start:batch_end]

    results.extend(get_dbpedia_same_as(cur_batch))

    batch_start += batch_size
    time.sleep(1)

    tqdm.write(f"Processed {batch_end} entities out of {num_entities}")

results = pd.DataFrame(results)
results = results.groupby(by=results["entity"]).agg(to_set)
results = results.reset_index()
results.to_csv(DB100K_PATH / 'mapping.csv', index=False, header=True)
