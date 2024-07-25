import pandas as pd

def row_to_string_with_dot(row):
    return f"{row["s"]} {row["p"]} {row["o"]} ."

train = pd.read_csv("./data/DB100K/mapped/train.txt", sep="\t", header=None, names=["s", "p", "o"])

train["s"] = train["s"].map(lambda s: f"<http://dbpedia.org/resource/{s}>")
train["p"] = train["p"].map(lambda p: f"<http://dbpedia.org/resource/{p}>")
train["o"] = train["o"].map(lambda o: f"<http://dbpedia.org/ontology/{o}>")

rows_with_dots = train.apply(row_to_string_with_dot, axis=1)

# Step 3: Write each row to a file
with open('./data/DB100K/mapped/train_rdf.txt', 'w') as file:
    for row in rows_with_dots:
        file.write(row + '\n')

from rdflib import Graph

rdflib_graph = Graph()
rdflib_graph.parse("./data/DB100K/mapped/train_rdf.txt", format="nt")

# SELECT ?s ?p ?o
# WHERE {
#     VALUES ?s { <http://dbpedia.org/resource/Boston> }
#     VALUES ?o { <http://dbpedia.org/resource/Kingdom_of_England> }
#     ?s ?p ?o .
# }
