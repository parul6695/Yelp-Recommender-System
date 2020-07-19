from pyspark import SparkConf, SparkContext, StorageLevel
from collections import Counter
from operator import add
import json
import os
import time
import sys
import itertools
import math
import csv
#import string
#from __future__ import print_function
import string

con = SparkConf().setAll([('spark.executor.memory', '8g'), ('master','local'),('appName','task1'),('spark.driver.memory','8g')])
sc = SparkContext(conf=con)
# variables

if len(sys.argv) != 4:
    print("Usage is incorrect")
    exit(-1)
else:
    input_file_path = sys.argv[1]
    #input_file_path = "train_review_task2.json"
    output_file_path = sys.argv[2]
    stopwords_file = sys.argv[3]
'''
#Example of sample input
input_file_path = r"train_review.json"
output_file_path = r"output_task2_train"
stopwords_file = r"stopwords"
'''
start = time.time()

stopWords = sc.textFile(stopwords_file).persist()
stopWords = stopWords.collect()
file = sc.textFile(input_file_path).persist()

def top200words(s, idf):
    # word frequencies
    word_freq = dict()
    for w in s:
        if w in word_freq:
            word_freq[w] += 1
        else:
            word_freq[w] = 1

    max_freq = max(word_freq.values())
    tf_idf = dict()

    for key, value in word_freq.items():
        tf = value / max_freq
        tf_idf[key] = tf * idf[key]
    k = Counter(tf_idf)
    top200 = k.most_common(200)
    return dict(top200)


def combineArray(array):
    s = ' '.join(array)
    s = s.translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits)).lower()
    swords = s.split()
    resultwords = [word for word in swords if word not in stopWords]
    s = ' '.join(resultwords)
    return s

bp1 = file.map(lambda x: (json.loads(x)['business_id'], json.loads(x)['text'])).groupByKey().mapValues(combineArray).map(lambda x: (x[0], x[1].split())).persist()

x = bp1.flatMap(lambda x: x[1]).count()
rare_word_limit = 0.000001 * x

words_lowcount = bp1.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b).filter(lambda x: x[1] < rare_word_limit).map(lambda x: x[0]).collect()
words_lowcount = set(words_lowcount)

def stringer2(swords):
    resultwords = [word for word in swords if word not in words_lowcount]
    return resultwords
bp = bp1.mapValues(stringer2).persist()

docs = bp.map(lambda x: x[0]).collect()
#print("==========================docs created")
total_docs = len(docs)

# Part b) Calculate TF-IDF for each business
def idfCal(ni):
    return math.log2(total_docs / ni)

def mapWordsToNums(d):
    #print("\n========== mapWordsToNum:")
    word_mapper = dict()
    counter = 0
    for key in d:
        word_mapper[key] = counter
        counter += 1
    return word_mapper

idf = bp.map(lambda x: (x[0], list(set(x[1])))).flatMap(lambda e: e[1]).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b).mapValues(idfCal).collectAsMap()  # [('good', 6), ('way', 1), ('inn', 1), ('type', 1), ('put', 1),.....]
word_mapper = mapWordsToNums(idf)
tf = bp.map(lambda x: (x[0], top200words(x[1], idf))).map(
    lambda x: (x[0], [word_mapper[x] for x in x[1]])).collectAsMap()
del idf
# Part e) User Profile

def findUnion(listOfDicts):
    return list(set.union(*listOfDicts))

up = file.map(lambda x: (json.loads(x)['user_id'], json.loads(x)['business_id'])).groupByKey().map(
    lambda x: (x[0], [y for y in x[1]])).map(lambda x: (x[0], findUnion([set(tf[x]) for x in x[1]]))).collectAsMap()

res = [tf, up]
with open(output_file_path, 'w', newline='')as f:
    for r in res:
        json.dump(r, f)
        f.write('\n')

end = time.time()
print("\n\nDuration:", end - start)
