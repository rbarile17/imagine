# Datasets

The datasets in this repository are DB50K, DB100K, YAGO4-20 as in [Kelpie++](https://github.com/rbarile17/kelpiePP).

Download the [datasets](https://zenodo.org/records/12819045)!

Each one includes:

- RDF triples in the files `train.txt`, `valid.txt`, `test.txt` where each line is a triple structured as follows:

  ```rdf
  subject'\t'predicate'\t'object
  ```

- entity classes in `entities.csv`
- the schema in one or more files depending on the KG
- the integration of the triples with the schema
- a `reasoned` directory containing:
  - the integrated dataset enriched after reasoning
  - the entity classes including implicit ones obtained through reasoning in `entities.csv`
