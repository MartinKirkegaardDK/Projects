from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import gensim
from tensorflow import keras
from tensorflow.keras import layers
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def make_list_of_lists(data):
    X = []
    Y = []
    first = True
    for elm in data.iterrows():
        if elm[1][0] == 1:
            if first != True:
                X.append(x)
                Y.append(y)
            x = []
            y = []
            first = False
        x.append(elm[1][1])
        y.append(elm[1][2])
    X.append(x)
    Y.append(y)
    return X, Y

def mask_out_for_balance(X, Y):
    mask = []
    for sentence in Y:
        if Counter(sentence)['O'] < len(sentence) / 2:
            mask.append(True)
        else:
            mask.append(False)
    Y_masked = np.array(Y)[mask]
    X_masked = np.array(X)[mask]
    
    return X_masked, Y_masked 

def encode_list_of_lists(X, max_length, count_vec):
    li = []

    for elm in X:
        t = []
        for i in range(max_length):
            if i >= len(elm):
                t.append(count_vec.transform(['asdfasdfasdfasdfasdfsaf']).toarray().flatten() == 1)
            else:
                t.append(count_vec.transform([elm[i]]).toarray().flatten() == 1)
        t = np.array(t)
        li.append(t)
    return np.array(li) 

def word2vec_list_of_lists(X, max_len, model):
    li = []
    
    for elm in X:
        t = []
        for i in range(max_len):
            if i >= len(elm):
                t.append(np.array([0]*model.vector_size).flatten())
            else:
                try: 
                    t.append(np.array(model[elm[i]]))
                except:
                    t.append(np.array([0]*model.vector_size).flatten())
        t = np.array(t)
        li.append(t)
    return np.array(li)  

def affix_list_of_lists(X, max_length, affix_transformer):
    li = []

    for elm in X:
        t = []
        for i in range(max_length):
            if i >= len(elm):
                t.append(np.zeros(affix_transformer.vector_size, dtype=bool))
            else:
                t.append(affix_transformer._map_affs(elm[i]))
        t = np.array(t)
        li.append(t)
    return np.array(li) 

def affix_list_of_lists_int(X, max_length, affix_transformer):
    li = []

    for elm in X:
        t = []
        for i in range(max_length):
            if i >= len(elm):
                t.append(np.zeros(len(affix_transformer.vector_size), dtype=int))
            else:
                t.append(affix_transformer._map_affs(elm[i]))
        t = np.array(t)
        li.append(t)
    return np.array(li).transpose((2,0,1))

lookup_table = {
    'O':0,
    'B-Negative': 1,
    'B-Positive': 2,
    'I-Negative': 3,
    'I-Positive':4,
    'B-Negative|I-Negative': 5,
    'B-Positive|I-Positive':6,
    'I-Positive|B-Positive': 7,
    'I-Positive|I-Positive': 8
}

def encode_labels(Y, lookup_labels, max_len):
    li = []
    for elm in Y:
        t = []
        counter = 0
        for i in range(max_len):
            if counter >= len(elm):
                t.append(np.zeros(9))
            else:
                one_hot = np.zeros(9)
                one_hot[lookup_labels[elm[i]]] = 1
                t.append(one_hot)
            counter += 1
        li.append(np.array(t))
    return np.array(li)

lookup_reverse = {
    0:'O',
    1:'B-Negative',
    2:'B-Positive',
    3:'I-Negative',
    4:'I-Positive',
    5:'B-Negative|I-Negative',
    6:'B-Positive|I-Positive',
    7:'I-Positive|B-Positive',
    8:'I-Positive|I-Positive'
}

def dump_pred(pred, data_raw, input_path, output_path, lookup): 
    comments = []
    with open(input_path) as f:
        for line in f.readlines():
            if line[0] == '#':
                comments.append(line)

    with open(output_path, 'w', encoding='UTF-8') as f:
        for sentence in range(len(comments)):
            f.write(comments[sentence])
            for word in range(len(pred[sentence])):
                f.write(str(word + 1) + '\t' + data_raw[sentence][word] + '\t' + lookup[pred[sentence][word]] + '\n')
            f.write('\n')

def make_predictions(model, X, sentence_lengths):
    pred = [[np.argmax(word) for word in sentence] for sentence in model.predict(X)]
    pred_no_padding = [labels[:length] for labels, length in zip(pred, sentence_lengths)]
    return pred_no_padding

def preprocess(data, encoding_type='onehot', w2v_path=None, model_w2v=None, count_vec=None, max_len=None):
    
    X_unencoded, Y_unencoded = make_list_of_lists(data)
    if not max_len:
        max_len = max([len(x) for x in X_unencoded])
    Y = encode_labels(Y_unencoded, lookup_table, max_len)
    
    if encoding_type == 'word2vec':
        if not w2v_path or model_w2v:
            print('Error: no w2v path2')
            return
        if not model_w2v:
            model_w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)
        X = word2vec_list_of_lists(X_unencoded, max_len, model_w2v)
        vector_size = model_w2v.vector_size
        return X, Y, vector_size,max_len, model_w2v
    elif encoding_type == 'onehot':
        if not count_vec:
            count_vec = CountVectorizer()
            count_vec.fit(data[1])
        X = encode_list_of_lists(X_unencoded, max_len, count_vec)
        vector_size = len(count_vec.get_feature_names_out()) 
        return X, Y, vector_size, max_len,count_vec     
    else:
        print('ERROR: invalid encoding_type')
        return

def get_max_len(data):
    return max([len(elm) for elm in data])
                  
def test_model(val_data, model, max_len, count_vec=None, model_w2v=None,encoding_type = 'onehot'):
    
    val_X, val_Y, _, _, _ = preprocess(val_data,encoding_type = encoding_type, count_vec = count_vec, max_len=max_len, model_w2v=model_w2v)
    print(val_X.shape)
    val_lengths = [len(sentence) for sentence in make_list_of_lists(val_data)[1]]
    
    val_pred = make_predictions(model, val_X, val_lengths)
    val_pred_flattened = sum(val_pred, [])
                  
    val_gt = [[np.argmax(word) for word in sentence] for sentence in val_Y]
    val_gt = [labels[:length] for labels, length in zip(val_gt, val_lengths)]
    val_gt_flattened = sum(val_gt, [])
                  
    print(classification_report(val_gt_flattened, val_pred_flattened))
    
def make_embedding_layers(train, output_dim=20, feature='affix'):
    layer_dict = dict()
    layer_dict['inputs'] = []
    layer_dict['embeddings'] = []
    for i in range(len(train.vector_size[feature])):
        layer_dict['inputs'].append(keras.Input(shape=(None,)))
        layer_dict['embeddings'].append(layers.Embedding(input_dim=train.vector_size[feature][i]+1, output_dim=output_dim, mask_zero=True)(layer_dict['inputs'][-1]))
    return layer_dict
    
class AffixRepresentation:
    def __init__(self):
        pass
    def fit(self, X, min_occurences=50, ngram_lengths=3):
        if type(ngram_lengths) == int:
            ngram_lengths = [ngram_lengths]
        pref_dict = defaultdict(lambda: 0)
        suf_dict = defaultdict(lambda: 0)
        for word in X:
            for ngram_length in ngram_lengths:
                if len(word) >= (2 * ngram_length):
                    pref_dict[word[:ngram_length]] += 1
                    suf_dict[word[-ngram_length:]] += 1
        self.prefs = dict()
        count = 0
        for pref in pref_dict.items():
            if pref[1] >= min_occurences:
                self.prefs[pref[0]] = count
                count += 1
                
        self.sufs = dict()
        count = 0
        for suf in suf_dict.items():
            if suf[1] >= min_occurences:
                self.sufs[suf[0]] = count
                count += 1
                
        self.ngram_lengths = ngram_lengths
        self.vector_size = len(self.prefs) + len(self.sufs)
                
    def _map_affs(self, word):
        oh_pref = np.zeros(len(self.prefs), dtype=bool)
        oh_suf = np.zeros(len(self.sufs), dtype=bool)
        for ngram_length in self.ngram_lengths:
                if len(word) >= (2 * ngram_length):
                    if word[:ngram_length] in self.prefs:
                        oh_pref[self.prefs[word[:ngram_length]]] = True
                    if word[-ngram_length:] in self.sufs:
                        oh_suf[self.sufs[word[-ngram_length:]]] = True
                        
        return np.concatenate((oh_pref, oh_suf))
                
    def transform(self, X):
        return X.map(self._map_affs)
        
class AffixRepresentationInt:
    
    def __init__(self):
        pass
    
    def fit(self, X, min_occurences=50, ngram_lengths=3):
        if type(ngram_lengths) == int:
            ngram_lengths = [ngram_lengths]
        pref_dict = {k:defaultdict(lambda: 0) for k in ngram_lengths}
        suf_dict = {k:defaultdict(lambda: 0) for k in ngram_lengths}
        
        for word in X:
            for ngram_length in ngram_lengths:
                if len(word) >= (2 * ngram_length):
                    pref_dict[ngram_length][word[:ngram_length]] += 1
                    suf_dict[ngram_length][word[-ngram_length:]] += 1
        
        self.prefs = {k:dict() for k in ngram_lengths}
        
        for k, v in pref_dict.items():
            count = 1
            for pref in v.items():
                if pref[1] >= min_occurences:
                    self.prefs[k][pref[0]] = count
                    count += 1
                
        self.sufs = {k:dict() for k in ngram_lengths}
        
        for k, v in suf_dict.items():
            count = 1
            for suf in v.items():
                if suf[1] >= min_occurences:
                    self.sufs[k][suf[0]] = count
                    count += 1
                
        self.ngram_lengths = ngram_lengths
        self.vector_size = [len(n) for n in self.prefs.values()] + [len(n) for n in self.sufs.values()]
        
    def _map_affs(self, word):
        pref = np.zeros(max(self.ngram_lengths), dtype=int)
        suf = np.zeros(max(self.ngram_lengths), dtype=int)
        for ngram_length in self.ngram_lengths:
            if len(word) >= (2 * ngram_length):
                if word[:ngram_length] in self.prefs[ngram_length]:
                    pref[ngram_length-1] = self.prefs[ngram_length][word[:ngram_length]]
                if word[-ngram_length:] in self.sufs[ngram_length]:
                    suf[ngram_length-1] = self.sufs[ngram_length][word[-ngram_length:]]
                        
        return np.concatenate(([pref[n-1] for n in self.ngram_lengths], [suf[n-1] for n in self.ngram_lengths]))
    
    def transform(self, X):
        return X.map(self._map_affs)
    
class Lemmatize_groups:
    def __init__(self, data):
        self.data = data
        self.wdl = WordNetLemmatizer()
        self.pos_list = self.pos_tag(self.data)
        self.lemmatized = None
        self.correct_format = {
            "NN":"n",
            "NNS": "n",
            "NNP": "n",
            "NNPS": "n",
            "jj":"a",
            "JJR": "a",
            "JJS": "a",    
            "VB": "v",
            "VBG": "v",
            "VBD": "v",
            "VBN": "v",
            "VBP": "v",
            "VBZ": "v",
            "RBR": "r",
            "RBS": "r",
            "RB": "r"
        }
        
    def pos_tag(self, X):
        pos_tag = []
        for i in X:
            tag = nltk.pos_tag(i)
            l = []
            for j in tag:
                l.append(j[1])
            pos_tag.append(l)
        return pos_tag
    
    def _lemmatize(self):
        li = []
        for sentences, pos_li in zip(self.data,self.pos_list):
            t = []
            for words, pos in zip(sentences, pos_li):
                if pos in self.correct_format:
                    t.append(self.wdl.lemmatize(words,pos = self.correct_format[pos]))
                else:
                    t.append(self.wdl.lemmatize(words))
            li.append(t)
        print("lemmatized done")
        self.lemmatized = li
    
    def fit_lemmatize(self, filter_amount = 50):
        if self.lemmatized == None:
            self._lemmatize()
        count_dict = defaultdict(lambda: 0)
        for sentence in self.lemmatized:  
            for word in sentence:
                count_dict[word] += 1
        
        final = dict()
        for key, val in count_dict.items():
            if val > filter_amount:
                final[key] = val
        self.group_dict = dict()
        counter = 2
        for key, val in final.items():
            self.group_dict[key] = counter
            counter += 1
        print("fit complete")
        
    def transform(self, obj):
        self.lemmatized_data = []
        if self.lemmatized == None:
            self._lemmatize()
        for sentence in self.lemmatized:
            t = []
            for word in sentence:
                if word in obj.group_dict:
                    t.append(obj.group_dict[word])
                else:
                    t.append(1)
            self.lemmatized_data.append(t)
        return self.lemmatized_data
    
    def fit_synonyms(self):
        #Dont use this
        self._lemmatize()
        total_set = set()
        d = dict()
        counter = 0
        for sentences in self.lemmatized:
            for words in sentences:
                s = set()
                syn = wordnet.synsets(words)

                if words.lower() not in total_set:
                    for elm in syn:
                        for i in elm.lemmas():
                            s.add(i.name().lower())
                            total_set.add(i.name().lower())
                d[counter] = s
                counter += 1
        final = dict()
        for key, val in d.items():
            for elm in list(val):
                final[elm] = key
        return final

class Preprocessor:
    def __init__(self, data):
        self.data = data
        self.X_unencoded, self.Y_unencoded = make_list_of_lists(self.data)
        
        self.max_len = max([len(x) for x in self.X_unencoded])
        self.vector_size = dict()
        
        
        
    def fit_one_hot(self):
        self.count_vec = CountVectorizer()
        self.count_vec.fit(self.data[1])
        self.vector_size['one_hot'] = len(self.count_vec.get_feature_names_out())
         
    def fit_affix(self, min_occurences=50, ngram_lengths=3):
        self.affix_model = AffixRepresentation()
        self.affix_model.fit(self.data[1], min_occurences, ngram_lengths)
        self.vector_size['affix'] = self.affix_model.vector_size
        
    def fit_affix_int(self, min_occurences=50, ngram_lengths=3):
        self.affix_model = AffixRepresentationInt()
        self.affix_model.fit(self.data[1], min_occurences, ngram_lengths)
        self.vector_size['affix'] = self.affix_model.vector_size
        
    def fit_word2vec(self, w2v_path):
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)
        self.vector_size['word2vec'] = self.w2v_model.vector_size
        
        
        
    def make_labels(self, train):
        return encode_labels(self.Y_unencoded, lookup_table, train.max_len)
        
    def make_one_hot(self, train):
        return encode_list_of_lists(self.X_unencoded, train.max_len, train.count_vec)
        
    def make_word2vec(self, train):
        return word2vec_list_of_lists(self.X_unencoded, train.max_len, train.w2v_model)
    
    def make_affix(self, train):
        return affix_list_of_lists(self.X_unencoded, train.max_len, train.affix_model)
    
    def make_affix_int(self, train):
        return affix_list_of_lists_int(self.X_unencoded, train.max_len, train.affix_model)
    
    
    def get_vector_size(self):
        return sum(self.vector_size.values())
    
    
    
    def test_model(self, X_val, Y_val, model):
        
        val_lengths = [len(sentence) for sentence in self.X_unencoded]
    
        val_pred = make_predictions(model, X_val, val_lengths)
        val_pred_flattened = sum(val_pred, [])

        val_gt = [[np.argmax(word) for word in sentence] for sentence in Y_val]
        val_gt = [labels[:length] for labels, length in zip(val_gt, val_lengths)]
        val_gt_flattened = sum(val_gt, [])
        
        return val_gt_flattened, val_pred_flattened


        
        
def padding(data, max_len):
    """Sven"""
    for elm in data:
        while len(elm) < max_len:
            elm.append(0)
    
    return np.array([np.array(x) for x in data])
        
