import numpy as np
import sys
import glob
import re
import os
import string
import pandas as pd
from collections import defaultdict
import json

def create_dataset(sources):
    """
    inputs a list of all filepaths
    outputs 2 lists:
    x1 - file content
    x1_paths - source path
    """
    x1=[]
    x1_paths=[]
    for src in sources:
        f = open(src, "r")
        file_content = f.read()[:-1]
        x1.append(file_content)
        x1_paths.append(src)
        f.close()
    return x1, x1_paths


valid_data="*/*/*/*.txt"
root=sys.argv[1]
path=os.path.join(root,valid_data)
valid_reviews=glob.glob(path)

# valid_reviews = glob.glob(valid_base_path,recursive=True)
valid_reviews = [review.replace("\\","/") for review in valid_reviews]
x,x_paths= create_dataset(valid_reviews)


# preprocess string function (stemming lementization stopwords to be added)

def preprocess_string(s):
    
    # words that should be removed (no contribution to prediction, computation)
    stop_words = [ 'are', 'around','as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before',
             'bottom', 'but', 'by', 'call', 'can', 'co', 'con', 'could',  'de', 'due', 'during', 'each', 'eg', 'else', 'etc', 'even','fill', 'find', 'fire', 'first', 'five', 'for','former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had',
             'has', 'hasnt', 'have', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed',
             'interest', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly',
             'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover',
             'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine',
             'no', 'nobody', 'none', 'noone', 'nor', 'not', 'now', 'nowhere', 'of', 'on', 'once','one', 'only',
              'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
             'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming',
             'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 
             'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system',
             't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',
             'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this',
             'those', 'though', 'three', 'through', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
             'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we',
             'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby',
             'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom',
             'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
             "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
             'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
             'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
             'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
             'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
            'then', 'once', 'here', 'there']
    
    s=s.translate(str.maketrans('', '', string.punctuation))
    s=re.sub('(\s+)',' ',s)
    s=s.lower()
    word_list = s.split(" ")
    new_list = []
    for word in word_list:
        if (len(word)>2) and (word not in stop_words):  
            new_list.append (word)
    s=" ".join(new_list)
    return s

# Test

classes=None
cats_info = None
def getExampleProb(test_example):                                

    likelihood_prob=np.zeros(classes.shape[0]) 

    for cat_index,cat in enumerate(classes): 

        for test_token in test_example.split():

            test_token_counts=cats_info[cat_index][0].get(test_token,0)+1

            test_token_prob=test_token_counts/float(cats_info[cat_index][2])                              

            likelihood_prob[cat_index]+=np.log(test_token_prob)

    post_prob=np.empty(classes.shape[0])
    for cat_index,cat in enumerate(classes):
        post_prob[cat_index]=likelihood_prob[cat_index]+np.log(cats_info[cat_index][1])                                  

    return post_prob


def test(test_set):

    predictions=[] 
    for example in test_set:                                 
        cleaned_example=preprocess_string(example)                  
        post_prob=getExampleProb(cleaned_example) 
        predictions.append(classes[np.argmax(post_prob)])

    return np.array(predictions) 

with open('nbmodel.txt') as weights:
    model_weights = json.load(weights)

classes= np.asarray(model_weights["tru_dec_info_classes"])
cats_info = np.asarray(model_weights["tru_dec_info"])
tru_dec_preds=test(x)
classes= np.asarray(model_weights["pos_neg_info_classes"])
cats_info = np.asarray(model_weights["pos_neg_info"])
pos_neg_preds=test(x)


my_file = open("nboutput.txt", "w")
for i in range(len(x_paths)):
    labela = None
    labelb = None
    if tru_dec_preds[i]==0:
        labela = "deceptive"
    else:
        labela = "truthful"
    if pos_neg_preds[i]==0:
        labelb = "negative"
    else:
        labelb = "positive"
            
    my_file.write(labela + " " + labelb + " " + x_paths[i] +  "\n")

my_file.close()
