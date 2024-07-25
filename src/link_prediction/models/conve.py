import copy
import torch
import numpy as np

from pydantic import BaseModel

from torch import nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

from .model import Model, KelpieModel
from ...dataset import Dataset, KelpieDataset


class ConvEHyperParams(BaseModel):
    dimension: int
    input_do_rate: float
    fmap_do_rate: float
    hid_do_rate: float
    hidden_layer_size: int


class ConvE(Model):
    def __init__(self, dataset: Dataset, hp: ConvEHyperParams, init_random=True):
        super().__init__(dataset)

        self.name = "ConvE"
        self.dataset = dataset
        self.num_entities = dataset.num_entities
        self.num_relations = 2 * dataset.num_relations

        self.dimension = hp.dimension
        self.input_dropout_rate = hp.input_do_rate
        self.feature_map_dropout_rate = hp.fmap_do_rate
        self.hidden_dropout_rate = hp.hid_do_rate
        self.hidden_layer_size = hp.hidden_layer_size

        self.embedding_width = 20
        self.embedding_height = self.dimension // self.embedding_width

        self.kernel_shape = (3, 3)
        self.num_filters = 32

        self.input_dropout = nn.Dropout(self.input_dropout_rate).cuda()
        self.feature_map_dropout = nn.Dropout2d(self.feature_map_dropout_rate).cuda()
        self.hidden_dropout = nn.Dropout(self.hidden_dropout_rate).cuda()
        self.batch_norm_1 = nn.BatchNorm2d(1).cuda()
        self.batch_norm_2 = nn.BatchNorm2d(self.num_filters).cuda()
        self.batch_norm_3 = nn.BatchNorm1d(self.dimension).cuda()
        conv_layer = nn.Conv2d(1, self.num_filters, self.kernel_shape, 1, 0, bias=True)
        self.convolutional_layer = conv_layer.cuda()
        self.hidden_layer = nn.Linear(self.hidden_layer_size, self.dimension).cuda()

        if init_random:
            entity_embs = torch.rand(self.num_entities, self.dimension).cuda()
            self.entity_embeddings = Parameter(entity_embs, requires_grad=True)
            relation_embs = torch.rand(self.num_relations, self.dimension).cuda()
            self.relation_embeddings = Parameter(relation_embs, requires_grad=True)
            xavier_normal_(self.entity_embeddings)
            xavier_normal_(self.relation_embeddings)

    def is_minimizer(self):
        return False

    def forward(self, triples):
        return self.all_scores(triples)

    def score(self, triples):
        all_scores = self.all_scores(triples)

        triples_scores = []
        for i, (_, _, o) in enumerate(triples):
            triples_scores.append(all_scores[i][o])

        return np.array(triples_scores)

    def score_embs(self, lhs, rel, rhs):
        lhs = lhs.view(-1, 1, self.embedding_width, self.embedding_height)
        rel = rel.view(-1, 1, self.embedding_width, self.embedding_height)

        stacked_inputs = torch.cat([lhs, rel], 2)
        stacked_inputs = self.batch_norm_1(stacked_inputs)
        stacked_inputs = self.input_dropout(stacked_inputs)

        feature_map = self.convolutional_layer(stacked_inputs)
        feature_map = self.batch_norm_2(feature_map)
        feature_map = torch.relu(feature_map)
        feature_map = self.feature_map_dropout(feature_map)
        feature_map = feature_map.view(feature_map.shape[0], -1)

        x = self.hidden_layer(feature_map)
        x = self.hidden_dropout(x)
        x = self.batch_norm_3(x)
        x = torch.relu(x)
        scores = torch.mm(x, rhs.transpose(1, 0))

        scores = torch.sigmoid(scores)
        output_scores = torch.diagonal(scores)

        return output_scores

    def criage_first_step(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        lhs = lhs.view(-1, 1, self.embedding_width, self.embedding_height)

        rel = self.relation_embeddings[triples[:, 1]]
        rel = rel.view(-1, 1, self.embedding_width, self.embedding_height)

        stacked_inputs = torch.cat([lhs, rel], 2)
        stacked_inputs = self.batch_norm_1(stacked_inputs)
        stacked_inputs = self.input_dropout(stacked_inputs)

        feature_map = self.convolutional_layer(stacked_inputs)
        feature_map = self.batch_norm_2(feature_map)
        feature_map = torch.relu(feature_map)
        feature_map = self.feature_map_dropout(feature_map)
        feature_map = feature_map.view(feature_map.shape[0], -1)

        x = self.hidden_layer(feature_map)
        x = self.hidden_dropout(x)
        x = self.batch_norm_3(x)
        x = torch.relu(x)

        return x

    def criage_last_step(self, x, rhs):
        scores = torch.mm(x, rhs.transpose(1, 0))

        scores = torch.sigmoid(scores)
        output_scores = torch.diagonal(scores)
        return output_scores

    def all_scores(self, triples):
        lhs = self.entity_embeddings[triples[:, 0]]
        lhs = lhs.view(-1, 1, self.embedding_width, self.embedding_height)

        rel = self.relation_embeddings[triples[:, 1]]
        rel = rel.view(-1, 1, self.embedding_width, self.embedding_height)

        stacked_inputs = torch.cat([lhs, rel], 2)
        stacked_inputs = self.batch_norm_1(stacked_inputs)
        stacked_inputs = self.input_dropout(stacked_inputs)

        feature_map = self.convolutional_layer(stacked_inputs)
        feature_map = self.batch_norm_2(feature_map)
        feature_map = torch.relu(feature_map)
        feature_map = self.feature_map_dropout(feature_map)
        feature_map = feature_map.view(feature_map.shape[0], -1)

        x = self.hidden_layer(feature_map)
        x = self.hidden_dropout(x)
        x = self.batch_norm_3(x)
        x = torch.relu(x)
        x = torch.mm(x, self.entity_embeddings.transpose(1, 0))

        pred = torch.sigmoid(x)

        return pred

    def predict_tails(self, triples):
        scores, ranks = [], []

        batch_size = 128
        for i in range(0, triples.shape[0], batch_size):
            batch = triples[i : min(i + batch_size, len(triples))]
            all_scores = self.all_scores(batch)
            objects = torch.tensor(batch[:, 2]).cuda()

            for j, (s, p, o) in enumerate(batch):
                o_to_filter = self.dataset.to_filter[(s, p)]
                target_o_score = all_scores[j, o].item()
                scores.append(target_o_score)

                all_scores[j, o_to_filter] = 0.0
                all_scores[j, o] = target_o_score

            _, sorted_indexes = torch.sort(all_scores, dim=1, descending=True)
            sorted_indexes = sorted_indexes.cpu().numpy()

            for row in range(batch.shape[0]):
                rank = np.where(sorted_indexes[row] == objects[row].item())[0][0]
                ranks.append(rank + 1)

        return scores, ranks

    def kelpie_model_class(self):
        return KelpieConvE

    def get_hyperparams_class():
        return ConvEHyperParams


class KelpieConvE(KelpieModel):
    def __init__(self, dataset: KelpieDataset, model: ConvE, init_tensor):
        hp = ConvEHyperParams(
            dimension=model.dimension,
            input_do_rate=model.input_dropout_rate,
            fmap_do_rate=model.feature_map_dropout_rate,
            hid_do_rate=model.hidden_dropout_rate,
            hidden_layer_size=model.hidden_layer_size,
        )
        self.model = ConvE(dataset, hp, init_random=False)
        self.original_entity = dataset.original_entity
        self.kelpie_entity = dataset.kelpie_entity

        frozen_entity_embs = model.entity_embeddings.clone().detach()
        frozen_relation_embs = model.relation_embeddings.clone().detach()

        self.kelpie_entity_emb = Parameter(init_tensor.cuda(), requires_grad=True)
        entity_embs = torch.cat([frozen_entity_embs, self.kelpie_entity_emb], 0)
        self.model.entity_embeddings = entity_embs
        self.model.relation_embeddings = frozen_relation_embs

        self.model.convolutional_layer = copy.deepcopy(model.convolutional_layer)
        self.model.convolutional_layer.weight.requires_grad = False
        self.model.convolutional_layer.bias.requires_grad = False
        self.model.convolutional_layer.eval()

        self.model.hidden_layer = copy.deepcopy(model.hidden_layer)
        self.model.hidden_layer.weight.requires_grad = False
        self.model.hidden_layer.bias.requires_grad = False
        self.model.hidden_layer.eval()

        self.model.batch_norm_1 = copy.deepcopy(model.batch_norm_1)
        self.model.batch_norm_1.weight.requires_grad = False
        self.model.batch_norm_1.bias.requires_grad = False
        self.model.batch_norm_1.eval()

        self.model.batch_norm_2 = copy.deepcopy(model.batch_norm_2)
        self.model.batch_norm_2.weight.requires_grad = False
        self.model.batch_norm_2.bias.requires_grad = False
        self.model.batch_norm_2.eval()

        self.model.batch_norm_3 = copy.deepcopy(model.batch_norm_3)
        self.model.batch_norm_3.weight.requires_grad = False
        self.model.batch_norm_3.bias.requires_grad = False
        self.model.batch_norm_3.eval()
