import pandas as pd

from tqdm import tqdm
from . import dataset, sparql

tqdm.pandas()


def get_domains_or_ranges(relation, to_get="domain"):
    relation_uri = f"<http://dbpedia.org/ontology/{relation}>"
    sparql.setQuery(
        f""" 
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?sub_result
        WHERE {{
            {relation_uri} rdf:type rdf:Property .
            {relation_uri} rdfs:{to_get} ?result .
            {{?sub_result rdfs:subClassOf ?result}}
        }}
        """
    )
    rows = sparql.queryAndConvert()["results"]["bindings"]
    return set([row["sub_result"]["value"] for row in rows])


relations = pd.DataFrame((dataset.relation_to_id.keys()), columns=["relation"])
relations["domains"] = relations["relation"].progress_map(get_domains_or_ranges)
relations["ranges"] = relations["relation"].progress_map(
    lambda relation: get_domains_or_ranges(relation, "range")
)

relations.to_csv("relations.csv", index=False, header=False)
