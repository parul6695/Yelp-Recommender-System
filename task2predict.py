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
import string
start = time.time()

con = SparkConf().setAll([('spark.executor.memory', '8g'), ('master','local'),('appName','task1'),('spark.driver.memory','8g')])
sc = SparkContext(conf=con)
# variables
'''
if len(sys.argv) != 4:
    print("Input not in form: $ spark-submit task1.py <input_file> <output_file>")
    exit(-1)
else:
    input_file_path = sys.argv[1]
    model=sys.argv[2]
    output_file_path = sys.argv[3]
'''
#Example of sample input
input_file_path = r"D:\DM sem2\Assignements\A3 Data\test_review.json"
model = r"D:\DM sem2\Assignements\A3 Data\output_task2_train"
output_file_path = r"D:\DM sem2\Assignements\A3 Data\output_task2_predict"

def calcCosineSim(user_id, business_id):
    if business_id not in calcSim:
        return 0
    if user_id not in calcSim:
        return 0
    b = set(calcSim[business_id])
    u = set(calcSim[user_id])
    return len(b.intersection(u)) / (math.sqrt(len(b)) * math.sqrt(len(u)))


with open(model,'r') as f:
    mod_file = [json.loads(line) for line in f]
calcSim = dict()
for m in mod_file:
    for key, value in m.items():
      calcSim[key] = value


output = sc.textFile(input_file_path).map(lambda x: (json.loads(x)['user_id'], json.loads(x)['business_id'])).map(lambda x: ((x[0], x[1]), calcCosineSim(x[0], x[1]))).filter(lambda x: x[1] >= 0.01).collect()

with open(output_file_path,'w') as f:
    for x in output:
        output_dict = {"user_id":x[0][0],"business_id":x[0][1],"sim":x[1]}
        json.dump(output_dict, f)
        f.write('\n')

end = time.time()
print("\n\nDuration:", end - start)