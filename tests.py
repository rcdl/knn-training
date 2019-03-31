# -*- coding:utf-8 -*-
from __future__ import print_function

import csv
import time

import numpy
from sklearn.model_selection import KFold
from matplotlib import pyplot

from trainer import SimpleKnn, WeightedKnn, AdaptativeKnn


def loadDatabase(document):
    with open(document, 'rb') as dotcsv:
        return list(csv.reader(dotcsv))


def get_db_items(db, idx):
    items = []

    for i in idx:
        items.append(db[i])

    return items


def get_result_schema(classifiers, ktries):
    results = {}

    for clsfy in classifiers:
        klass = clsfy.__name__
        results[klass] = {
            'training_time_samples': [],
            'kmodes': {}
        }
        for k in ktries:
            results[klass]['kmodes']['{k}-nn'.format(k=k)] = {
                'kdx': k,
                'testing_time_samples': [],
                'accuracy_samples': [],
            }

    return results


def test_run_v2(classifier, testdb, runsults):
    correto = 0.0
    total = len(testdb)

    started  = time.time()
    for item in testdb:
        if classifier.classify(item) == item[-1]:
            correto += 1
    runsults['testing_time_samples'].append(time.time() - started)
    runsults['accuracy_samples'].append(correto*100.0/total)


def report_results(results):
    # Training time
    names, values, yerr = [], [], []
    for key, result in results.iteritems():
        names.append(key)
        values.append(numpy.average(result['training_time_samples'])*1000)
        yerr.append(numpy.std(result['training_time_samples']))
    
    fig, ax = pyplot.subplots()
    fig.suptitle("Training time per classifier")
    ax.scatter(names, values)
    ax.errorbar(names, values, yerr=yerr, linestyle="None")
    ax.set_ylabel("Training time (ms)")
    ax.set_ylim(bottom=0)
    pyplot.show()

    # Test time
    fig, ax = pyplot.subplots()
    fig.suptitle("Testing time per k-classifier ")
    for key, result in results.iteritems():
        idx,avg, std = [], [], []
        for knn, samples in sorted(result['kmodes'].iteritems(), key=lambda x: x[1]['kdx']):
            idx.append(samples['kdx'])
            avg.append(numpy.average(samples['testing_time_samples'])*1000)
            std.append(numpy.std(samples['testing_time_samples'])*1000)
        ax.scatter(idx, avg, label=key)
        ax.errorbar(idx, avg, yerr=std, linestyle="None")
    ax.set_ylabel("Training time (ms)")
    ax.set_ylim(bottom=0)
    ax.legend()
    pyplot.show()

    # Accuracy
    fig, axs = pyplot.subplots(ncols=3,sharey=True)
    fig.suptitle("Accuracy per k-classifier ")
    iplot=0
    colors=['r','g','b']
    for key, result in results.iteritems():
        idx, avg, std = [], [], []
        for knn, samples in sorted(result['kmodes'].iteritems(), key=lambda x: x[1]['kdx']):
            idx.append(samples['kdx'])
            avg.append(numpy.average(samples['accuracy_samples']))
            std.append(numpy.std(samples['accuracy_samples']))
        axs[iplot].scatter(idx, avg, c=colors[iplot])
        axs[iplot].errorbar(idx, avg, c=colors[iplot], yerr=std, linestyle="None")
        axs[iplot].set_ylabel("Accuracy (%)")
        axs[iplot].set_xlabel(key)
        axs[iplot].set_ylim(top=100)
        iplot += 1
    pyplot.show()


def run_tests(db_name, n_splits):
    db = loadDatabase(db_name)
    ms = KFold(n_splits=n_splits, shuffle=True)
    db_folds = []
    classifiers = [SimpleKnn, WeightedKnn, AdaptativeKnn]
    kmodes = [1,2,3,5,7,9,11,13,15]
    results = get_result_schema(classifiers, kmodes)

    for train, test in ms.split(db):
        trainset = get_db_items(db, train)
        testset = get_db_items(db, test)
        for Classifier in classifiers:
            reskey = Classifier.__name__
            print(reskey, end='')
            for k in kmodes:
                knn = Classifier(k)
                results[reskey]['training_time_samples'].append(knn.train(trainset))
                test_run_v2(knn, testset, results[reskey]['kmodes']['{k}-nn'.format(k=k)])
                print('.', end='')
            print('Done')

    report_results(results)


def run_main():
    db_table = [
        ('datatrieve.csv', 5),
        ('kc2.csv', 10)
    ]

    for filename, nfolds in db_table:
        print("Running tests for db {db}".format(db=filename))
        run_tests(filename, nfolds)


if __name__ == '__main__':
    run_main()

