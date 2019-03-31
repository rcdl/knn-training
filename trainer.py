# -*- coding:utf-8 -*-
import math

from refenrece import carregarDados


class SimpleKnn(object):
    training_instances = None

    def __init__(self, instances):
        self.train(instances)

    def train(self, instances):
        self.training_instances = instances

    def get_distance(self, target, ref):
        acc = 0

        for i in range(len(ref)):
            acc += pow((float(ref[i]) - float(target[i])), 2)

        return math.sqrt(acc)

    def get_neighbors(self, k, target):
        neighbors = []

        for item in self.training_instances:
            attributes = item[:-1]
            neighbors.append((item, self.get_distance(target, attributes)))

        return sorted(neighbors, key=lambda x: x[1])[:k]

    def classify(self, target, k):
        dist = {}

        for item in self.get_neighbors(k, target):
            klass = item[0][-1]
            if klass in dist:
                dist[klass] += 1
            else:
                dist[klass] = 1

        return sorted(dist, key=lambda x: dist[x]).pop()


class WeightedKnn(SimpleKnn):

    def classify(self, target, k):
        dist = {}

        for attr, weight in self.get_neighbors(k, target):
            klass = attr[-1]
            if klass in dist:
                dist[klass] += 1/pow(weight+0.001, 2)
            else:
                dist[klass] = 1/pow(weight+0.001, 2)

        return sorted(dist, key=lambda x: dist[x]).pop()


class AdaptativeKnn(SimpleKnn):
    pass
