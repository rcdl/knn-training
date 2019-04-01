# -*- coding:utf-8 -*-
import math
import time


class SimpleKnn(object):
    training_instances = None
    k = None

    def __init__(self, k=1):
        self.k = k
        self.training_instances = []

    def train(self, instances):
        self.training_instances = self.preprocess(instances)
        return 0.0

    def preprocess(self, instances):
        processed = []

        for instance in instances:
            processed.append({
                'kls': instance[-1],
                'attr': instance[:-1]
            })

        return processed

    def get_distance(self, target, ref):
        acc = 0

        for i in range(len(ref)):
            acc += pow((float(ref[i]) - float(target[i])), 2)

        return math.sqrt(acc)

    def get_neighbors(self, target):
        neighbors = []

        for item in self.training_instances:
            attributes = item['attr']
            neighbors.append((item, self.get_distance(target, attributes)))

        return sorted(neighbors, key=lambda x: x[1])[:self.k]

    def classify(self, target):
        dist = {}

        for item in self.get_neighbors(target):
            klass = item[0]['kls']
            if klass in dist:
                dist[klass] += 1
            else:
                dist[klass] = 1

        return sorted(dist, key=lambda x: dist[x]).pop()


class WeightedKnn(SimpleKnn):

    def classify(self, target):
        dist = {}

        for attr, weight in self.get_neighbors(target):
            klass = attr['kls']
            if klass in dist:
                dist[klass] += 1/pow(weight, 2)
            else:
                dist[klass] = 1/pow(weight, 2)

        return sorted(dist, key=lambda x: dist[x]).pop()


class AdaptativeKnn(SimpleKnn):
    
    def train(self, instances):
        super(AdaptativeKnn, self).train(instances)
        started = time.time()

        for instance in self.training_instances:
            instance['atkrad'] = self.get_attack_radius(instance, self.training_instances)

        return time.time() - started

    def get_attack_radius(self, instance, instances):
        enemies = []

        for item in instances:
            if item['kls'] != instance['kls']:
                enemies.append(self.get_distance(instance['attr'], item['attr']))

        return sorted(enemies, reverse=True).pop()


    def get_neighbors(self, target):
        neighbors = []

        for item in self.training_instances:
            attributes = item['attr']
            neighbors.append((item, self.get_distance(target, attributes)/item['atkrad']))

        return sorted(neighbors, key=lambda x: x[1])[:self.k]

