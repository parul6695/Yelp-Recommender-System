from time import time
from pyspark import SparkConf, SparkContext,StorageLevel
import sys
import json
import itertools
import random
con = SparkConf().setAll([('spark.executor.memory', '8g'), ('master','local'),('appName','task1'),('spark.driver.memory','8g')])
sc = SparkContext(conf=con)
jaccard_support = 0.05
num_hashes = 64
n_bands = 64
n_rows = 1
res=[]
start = time()

if len(sys.argv) != 3:
    print("Input not in form: $ spark-submit task1.py <input_file> <output_file>")
    exit(-1)
else:
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
'''
#sample input
inputFile=r"D:\DM sem2\Assignements\A3 Data\train_review.json"
outputFile=r"D:\DM sem2\Assignements\A3 Data\output_task1"
print(inputFile)
'''
#json.loads as a function to apply to each line of text file to convert string to a python dictionary
reviewRdd = sc.textFile(inputFile).map(json.loads)
reviewRdd1=reviewRdd.map(lambda e : (( e['user_id'],e['business_id'],e['stars']))).persist(storageLevel=StorageLevel.MEMORY_ONLY)
print("hiii")
print(reviewRdd1.take(2))
users = reviewRdd1.map(lambda entry: entry[0]).distinct()
num_users = users.count()
#generating SEQUENCE numbers using zipWithIndex
#collectAsMap will return the results for paired RDD as Map collection.
#And since it is returning Map collection you will only get pairs with unique
#keys and pairs with duplicate keys will be removed.
user_index_dict = reviewRdd1.map(lambda entry: entry[0]).zipWithIndex().collectAsMap()
#grouping all users against the business ID that have put review against that business
business_user_map = reviewRdd1\
    .map(lambda entry: (entry[1], user_index_dict.get(entry[0]))).groupByKey().sortByKey()
bMapValues=business_user_map.mapValues(lambda entry: set(entry)).collect()
business_char_matrix = {}
for bu in bMapValues:
    business_char_matrix.update({bu[0]:bu[1]})

def applyLSHToSignature(business_id, signatures, n_bands, n_rows):
    signature_tuples = []
    for band in range(0,n_bands):
        band_name = band
        final_signature = signatures[band*n_rows:(band*n_rows)+n_rows]
        final_signature.insert(0, band_name)
        signature_tuple = (tuple(final_signature), business_id)
        signature_tuples.append(signature_tuple)
    return signature_tuples

def minhashArray(users,num_hashes, num_users):
    users = list(users)
    system_max_value = sys.maxsize
    hashed_users = [system_max_value for i in range(0,num_hashes)]
    #random_a = list(random_coeffs.get('a'))
    #random_b = list(random_coeffs.get('b'))
    for user in users:
        for i in range(1, num_hashes+1):
            current_hash_code = ((i*user)+ (67*i*71)) % num_users
            if current_hash_code < hashed_users[i-1]:
                hashed_users[i-1] = current_hash_code
    return hashed_users

def generate_similar_businesses(businesses):
    b_length = len(businesses)
    similar_businesses = []
    similar_businesses = sorted(list(itertools.combinations(sorted(businesses),2)))
    return similar_businesses

candidates = business_user_map\
    .mapValues(lambda users: minhashArray(users, num_hashes, num_users))
candidates2=candidates.flatMap(lambda entry: applyLSHToSignature(entry[0], list(entry[1]), n_bands, n_rows)).groupByKey().filter(lambda entry: len(list(entry[1])) > 1).flatMap(lambda entry: generate_similar_businesses(sorted(list(entry[1]))))\
    .distinct()\
    .persist()
#The value sys.maxsize, on the other hand, reports the platform's pointer size, and that limits the size of
#Python's data structures such as strings and lists.
def calculate_jaccard(candidate, business_char_matrix):
    users_c1 = set(business_char_matrix.get(candidate[0]))
    users_c2 = set(business_char_matrix.get(candidate[1]))
    jaccard_intersection = len(users_c1.intersection(users_c2))
    jaccard_union = len(users_c1.union(users_c2))
    jaccard_similarity_value = float(jaccard_intersection)/float(jaccard_union)
    #print("candidate, jaccard_similarity_value",candidate, jaccard_similarity_value)
    return (candidate,jaccard_similarity_value)

final_pairs = candidates2\
    .map(lambda cd: calculate_jaccard(cd, business_char_matrix)).filter(lambda cd: cd[1] >= jaccard_support)\
    .sortByKey()
result = final_pairs.collect()
with open(outputFile, "w+") as fp:
    a=[]
    for x in result:
        final_dic={"b1":x[0][0],"b2":x[0][1],"sim":x[1]}
        json.dump(final_dic, fp)
        fp.write('\n')

end = time()
print("Count: ", len(result))
print("Duration: " + str(end-start))
