# Imagine

Imagine is a post-hoc local explainability tool for Link Prediction (LP) on Knowledge Graphs (KGs) through embedding-based models. It explains a prediction $\langle s, p, o\rangle$ by identifying the smallest set of facts that enabled such inference.

It provides _Additive Counterfactual Explanations_ (ACEs) consisting of _additional triples_. The _additional triples_ are those triples that are neither explicitly stated in, nor entailed by, the KG, yet are not assumed to be false under _Open World Assumption_. It explains a prediction $\langle s, p, o\rangle$ by identifying the smallest set of facts that enabled such inference. Specifically, given a true triple $t = \langle s, p, o\rangle$, Imagine returns as explanation a set of _additional triples_ featuring $s$ that is relevant for $t$, i.e., leading the LP method to modify its rank.

## Architecture

Imagine is structured on four components:

* **Triple Builder**: generates additional triples featuring _s_
* **Pre-Filter**: selects the most useful additional triples
* **Explanation Builder**: combines the pre-filtered triples into candidate explanations and identifies sufficiently relevant ones
* **Relevance Engine**: estimates the _relevance_ of a candidate explanation

## Additional information

Check also:

* [models README](./models/README.md)
* [data README](./data/README.md)

## Getting started

```bash
# Clone the repository
git clone https://github.com/rbarile17/imagine.git

# Navigate to the repository directory
cd imagine

# Install the required dependencies
pip install -r requirements.txt
```

## Usage Instructions

Follow the steps in this section to run the pipeline.
All commands require the parameters `dataset`, `model`.
Find in `data` the datasets DB50K, DB100K, and YAGO4-20.
You can also experiment with your own datasets! (structured as explained in data [README](./data/README.md))
Instead, the supported models are: ComplEx, ConvE, TransE.
You can extend the class `Model` to add models!

Run the commands with the --help option to inspect the possible values for all the parameters!

### Preliminary

Create a `<model>_<dataset>.json` in `configs` specifying the config for training, explanation, and evaluation. Check out [configs README](./configs/README.md) for information and examples on configurations.

### Link Prediction

#### Train a model

```python
python -m src.link_prediction.train --dataset <dataset> --model <model> --valid <validation_epochs>
```

`<valid>` is the frequency (in epochs) of evaluation of the model on the validation set to determine whether to apply early stopping

#### Make **predictions** (compute the rank of test triples)

```python
python -m src.link_prediction.test --dataset <dataset> --model <model>
```

#### Select and sample 100 predictions

```python
python -m src.selec_preds --dataset <dataset> --model <model> --pred-rank <pred_rank>
```

<pred_rank> specifies which predictions to select based on their rank, choose between:

* `any`
* `first`
* `notfirst`

### Explanation and Evaluation

The commands in this section also require:

* `<method>`: the explanation method, choose between:
  * `imagine` (default)
  * `i-kelpie`: baseline method for the experimental evaluation, `i-kelpie++` is its enhancement featuring summarization in the **Explanation Builder**
  * [`kelpie`](https://github.com/AndRossi/Kelpie) and [`kelpie++`](https://github.com/rbarile17/kelpiePP) as baslines for the experimental evaluation
  * `wimagine`: baseline method for the experimental evaluation
* `<summarization>` (to specify solely if the method is one of `imagine`, `wimagine`, `ikelpie++`, `kelpie++`) is the summarization solution to adopt in the **Explanation Builder**, choose between the following values (ordered by increasing granularity):
  * `simulation`
  * `bisimulation`
  * `no` (default)

#### Generate explanations

```python
python -m src.explain --method <method> --dataset <dataset> --model <model> --mode <mode> --summarization <summarization>
```

#### Assses the impact of explanations on ranks

```python
python -m src.evaluation.verify_explanations --method <method> --dataset <dataset> --model <model> --mode <mode> --summarization <summarization>
```

#### Compute evaluation metrics

```python
python -m src.evaluation.compute_metrics --method <method> --dataset <dataset> --model <model> --mode <mode> --summarization <summarization>
```

## Experiments

To reproduce the experiments in the paper use:

* the datasets [DB50K](./data/DB50K), [DB100K](./data/DB100K), [YAGO4-20](./data/YAGO4-20)
* our [configs](./experiments/new/.configs) specifying the hyperparameters found as described in Appendix B of the paper
* our [pre-trained models](https://zenodo.org/records/11452683)
* our [sampled correct preds](./experiments/new/.selected_preds)

### Hyper-parameters

We report the hyper-parameters that we adopted in all phases of the experimental evaluation.

| **Model**   | **Parameter** | **DB50K** | **DB100K** | **YAGO4-20** |
|-------------|---------------|-----------|------------|--------------|
| **TransE**  | $D$           | 64        | 64         | 6            |
|             | $p$           | 2         | 1          | 2            |
|             | $Ep$          | 60        | 165        | 45           |
|             | $Lr$          | 0.003     | 0.002      | 0.042        |
|             | $\gamma$      | 10        | 2          | 2            |
|             | $N$           | 5         | 15         | 10           |
| **ConvE**   | $D$           | 200       | 200        | 200          |
|             | $Drop.in$     | 0.1       | 0          | 0.2          |
|             | $Drop.h$      | 0         | 0.1        | 0            |
|             | $Drop.feat$   | 0         | 0.2        | 0            |
|             | $Ep$          | 65        | 210        | 210          |
|             | $Lr$          | 0.030     | 0.013      | 0.007        |
| **ComplEx** | $D$           | 256       | 256        | 256          |
|             | $Ep$          | 39        | 259        | 149          |
|             | $Lr$          | 0.046     | 0.029      | 0.015        |

Note that:

* $D$ is the embedding dimension, in the models that we adopted entity and relation embeddings always have the same dimension
* $p$ is the exponent of the $p$-norm
* $Lr$ is the learning rate
* $B$ is the batch size
* $Ep$ is the number of epochs
* $\gamma$ is the margin in the _Pairwise Ranking Loss_
* $N$ is the number of negative triples generated for each positive triple
* $\omega$ is the size of the convolutional kernels
* $Drop$ is the training dropout rate, specifically:
  * $in$ is the input dropout
  * $h$ is the dropout applied after a hidden layer
  * $feat$ is the feature dropout

We adopted _Random Search_ to find the values of the hyper-parameters. Exceptions are given by $B$ and $Ep$. Specifically, for $B$ we adopted the value $8192$ for all configurations as it leads to optimize execution times and parallelism. While, for $Ep$ we adopted early stopping with $1000$ as maximum number of epochs, $5$ as patience threshold, and evaluating the model on the validation set every $5$ epoch during the training of the models. Then, we reported the epoch on which the training stopped. Hence, we used such value as number of epochs in the _post-training_ and in the re-training during evaluation.
Furthermore, we adopted the learning rate ($Lr$) values in the table during training and evaluation, but for the _post-training_ we used a different value. The batch size ($B$) is particularly large (8192) and usually exceeds by far the number of triples featuring an entity. This affects _post-training_ because in any _post-training_ epochs the entity would only benefit from one optimization step. We easily balanced this by increasing the $Lr$ to $0.01$.

## Repository Structure

  ├── README.md          <- The top-level README for developers using this project.
  ├── data
  ├── notebooks          <- Jupyter notebooks.
  ├── requirements.txt   <- The requirements file for reproducing the environment
  │
  └── src                <- Source code.
