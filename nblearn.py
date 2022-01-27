import numpy as np
import glob
import sys
import os
import re
import string
import pandas as pd
from collections import defaultdict
import json

def create_dataset(sources):
    """
    inputs a list of all filepaths
    outputs 4 lists:
    x1 - file content
    y1 - label corresponding to x1 (positive/negative)
    x2 - file content
    y2 - label corresponding to x2 (truthful/deceptive)
    """
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for src in sources:
        f = open(src, "r")
        file_content = f.read()[:-1]
        x1.append(file_content)
        x2.append(file_content)
        f.close()
        if src.split("/")[-4]=="negative_polarity":
            y2.append(0)
        else:
            y2.append(1)
        if src.split("/")[-3]=="deceptive_from_MTurk":
            y1.append(0)
        else:
            y1.append(1)
    return x1,y1,x2,y2


# creating training datasets

train_data="*/*/*/*.txt"
root=sys.argv[1]
path=os.path.join(root,train_data)
train_reviews=glob.glob(path)

train_reviews = [review.replace("\\","/") for review in train_reviews]

x1_train,y1_train,x2_train,y2_train = create_dataset(train_reviews)

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

# training

classes=None
bow_dicts=None

def addToBow(example,dict_index):
    global bow_dicts

    if isinstance(example,np.ndarray): example=example[0]
    for token_word in example.split():
        bow_dicts[dict_index][token_word]+=1

def train(dataset,labels):
    global bow_dicts, classes
    
    examples=dataset
    labels=labels
    if not isinstance(examples,np.ndarray): examples=np.array(examples)
    if not isinstance(labels,np.ndarray): labels=np.array(labels)
        
    classes = np.unique(labels)
    bow_dicts=np.array([defaultdict(lambda:0) for index in range(classes.shape[0])])

    for cat_index,cat in enumerate(classes):

        all_cat_examples=examples[labels==cat] 

        cleaned_examples=[preprocess_string(cat_example) for cat_example in all_cat_examples]
        cleaned_examples=pd.DataFrame(data=cleaned_examples)

        np.apply_along_axis(addToBow,1,cleaned_examples,cat_index)


    prob_classes=np.empty(classes.shape[0])
    all_words=[]
    cat_word_counts=np.empty(classes.shape[0])
    for cat_index,cat in enumerate(classes):

        prob_classes[cat_index]=np.sum(labels==cat)/float(labels.shape[0]) 

        count=list(bow_dicts[cat_index].values())
        cat_word_counts[cat_index]=np.sum(np.array(list(bow_dicts[cat_index].values())))+1

        all_words+=bow_dicts[cat_index].keys()

    vocab=np.unique(np.array(all_words))
    vocab_length=vocab.shape[0]

    denoms=np.array([cat_word_counts[cat_index]+vocab_length+1 for cat_index,cat in enumerate(classes)])                                                                          

    cats_info=[(bow_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(classes)]                               
    cats_info=np.array(cats_info) 
    
    return cats_info

model_weights = {}

tru_dec_info = train(x1_train,y1_train)
model_weights["tru_dec_info_classes"] = classes.tolist()
model_weights["tru_dec_info"] = tru_dec_info.tolist()

classes=None
bow_dicts=None
pos_neg_info = train(x2_train,y2_train)
model_weights["pos_neg_info_classes"] = classes.tolist()
model_weights["pos_neg_info"] = pos_neg_info.tolist()


with open('nbmodel.txt', 'w') as outfile:
    json.dump(model_weights, outfile)