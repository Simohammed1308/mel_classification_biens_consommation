## Code TF KERAS IA PYTHON 3 - compagnon de l'ouvrage des éditions ENI nommé :

## TensorFlow et Keras
## L’intelligence artificielle appliquée à la robotique humanoïde

## Auteur du code et de l'ouvrage : Henri Laude 
## 2019

## Aucune responsabilité d'aucune sorte ne peut être affectée à l'auteur de code
## ou aux éditions ENI pour un quelconque usage que vous pourriez en faire

## Ce code vous est procuré sous la licence opensource creativ commons de type : CC-BY-NC

## Le code est classé dans son ordre d'appartition dans l'ouvrage
## il convient de recopier l'extrait que vous voulez étudier dans votre EDI python
## Evidemment son exécution est conditionnée au fait que vous ayez installé les packages nécessaires
## et que vous ayez le cas échéant créé des données en entrée l'extrait  de code testé le nécessite

## les extraits son référencés par chapitre de l'ouvrage et par ordre séquentiel


##  ch02 ####################### extrait de code 1


# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

#' Ce code suppose que Tensorflow 2 est installé, si ce n'est pas le cas :
#'
#' 1) installer python 3.x et s'assurer qu'il est dans le Path
#' 2) intaller pip s'il n'est pas installé :
#'
#' curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#' python get-pip.py
#' 
#' 3) installer tensorflow, matplotlib et data set d'exemple
#' pip install tensorflow==2.0.0-alpha0
#' pip install matplotlib
#' pip install tensorflow-datasets


# TensorFlow and tf.keras
import tensorflow as tf

# Vérification que l'on dispose des compagnons de Tensorflow
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display_html

import time



import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import words, stopwords
from string import punctuation


from sklearn import manifold, decomposition
from sklearn import cluster, metrics

from transformers import AutoTokenizer

#from nltk import word_tokenize          
#import spacy

from wordcloud import WordCloud
from PIL import Image

import plotly as px

from pandarallel import pandarallel


# Si tout se passe bien, il n'y pas d'erreur et vous visualisez la version
# de Tensorflow que vous avez installée
#print("Version de tensorflow :",tf.__version__)

#print("Mode exécution immédiate : ",tf.executing_eagerly())
print("version pandas : ", pd.__version__)

def formats(dataframe, name):
    print("version pandas : ", pd.__version__)
    formats = pd.DataFrame([dataframe.shape],
                       columns=['Nbre de lignes','Nbre de variables'],
                       index=[name])
    return formats

def get_types_objects(df):
    df_object = pd.DataFrame()
    df_float = pd.DataFrame()
    df_int = pd.DataFrame()
    df_bool = pd.DataFrame()
    for col in df.columns:
        if ((df[col].dtypes == 'object')):
            df_object[col] = df[col]
        elif (df[col].dtypes == 'int64'):
            df_int[col] = df[col]
        elif((df[col].dtypes == 'bool')):
            df_bool[col] = df[col]
        else:
            df_float[col] = df[col]
            
    return df_object, df_int, df_float,df_bool

def colunmLigneDuplicated(dataframe, name):
    a = dataframe.columns.duplicated().sum()
    b = dataframe.duplicated().sum()
    duplicated = pd.DataFrame([(str(a),str(b))],
                       columns=['Colonnes dupliquées','Lignes dupliquées'],
                       index=[name])
    return duplicated


def vars_types(df):
    df_objet, df_int, df_float, df_bool = get_types_objects(df)
    types = {'Objet':df_objet.shape[1],
        'Float':df_float.shape[1],
        'Int':df_int.shape[1],
        'Bool':df_bool.shape[1]
    }
    return pd.DataFrame([types.values()], columns=types.keys(),index=[''])

def display_dfs(dfs, gap=50, justify='center'):
    html = ""
    for title, df in dfs.items():  
        df_html = df._repr_html_()
        cur_html = f'<div> <h3>{title}</h3> {df_html}</div>'
        html +=  cur_html
    html= f"""
    <div style="display:flex; gap:{gap}px; justify-content:{justify};">
        {html}
    </div>
    """
    display_html(html, raw=True)

def dessinerCamembert(df, col):
    plt.figure(figsize=(20,8))

    colors = sns.color_palette('bright')[0:5]
    plt.title('Répartition des '+col+' en %', size=20)
    wedges, texts, autotexts = plt.pie(df[col].value_counts().values, 
            labels = df[col].value_counts().index.str.upper(),
           autopct='%1.1f%%', textprops={'fontsize': 16 } , colors = colors)


    ax = plt.gca()

    ax.legend(wedges, df[col].value_counts().index.str.upper(),
              title=col,
              loc="upper left",
              fontsize=14,
              bbox_to_anchor=(1, 0, 0.5, 1))
    #fct_exp.save_fig("repartition_grades_nutriscores_perc")



    plt.figure(figsize=(20,8))

    sns.set_theme(style="whitegrid")
    sns.countplot(x=df[col], order=df[col].value_counts().index)
    plt.title('Répartition des '+col, size=20)
    #fct_exp.save_fig("repartition_grades_nutriscores_count")
    plt.show()

def display_tokens_info(tokens):
    """
    display info about corpus
    """
    print(f"nb tokens {len(tokens)}, nb tokens unique {len(set(tokens))}")
    print(tokens[:30])    
    
    
def process_text(doc, rejoin=False, lemm_or_stem=None, list_rare_words=None,  list_more_words=None,
                   min_len_word=3, force_is_alpha=True, eng_words=None, extra_words=None):

    
    # list unique words
    if not list_rare_words:
        list_rare_words = []
    
    # list more words
    if not list_more_words:
        list_more_words = []
    
    # extra_words
    if not extra_words:
        extra_words = []
    
    # lower and strip
    doc = doc.lower().strip()
    
    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)
    
    # classics stopwords
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stopwords_combined]
    
    ###############################################################
    ###############################################################
    
    # no rare tokens
    non_rare_tokens = [w for w in cleaned_tokens_list if w not in list_rare_words]
    
    
    ###############################################################
    ###############################################################
    
    # no more tokens
    non_more_tokens = [w for w in non_rare_tokens if w not in list_more_words]
    
    # no more len words
    more_than_N = [w for w in non_more_tokens if len(w) >= min_len_word]
    
    # only alpha chars
    if force_is_alpha :
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else :
        alpha_tokens = more_than_N
        
    ####################################################################
    ####################################################################
    
    # stem or lemm
    if lemm_or_stem == 'lem' :
        trans = WordNetLemmatizer()
        trans_text = [trans.lemmatize(i) for i in alpha_tokens]
    else : 
        trans = PorterStemmer()
        trans_text = [trans.stem(i) for i in alpha_tokens]
    
    ####################################################################
    ####################################################################
    
    # in english
    if eng_words :
        engl_text = [i for i in trans_text if i in eng_words]
    else : 
        engl_text = trans_text
    
    ####################################################################
    ####################################################################
    
    # manage return type
    if rejoin :
        return " ".join(engl_text)
    return engl_text    
    
def process_text(doc, rejoin=False, min_len_word=3, force_is_alpha=True, eng_words=None, extra_words=None):

    
    # extra_words
    if not extra_words:
        extra_words = []
    
    # lower and strip
    doc = doc.lower().strip()
    
    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)
    
    # classics stopwords
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stopwords_combined]
    
    ###############################################################
    ###############################################################
    
    # no more len words
    more_than_N = [w for w in cleaned_tokens_list if len(w) >= min_len_word]
    
    # only alpha chars
    if force_is_alpha :
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else :
        alpha_tokens = more_than_N
        
    ####################################################################
    ####################################################################
    
    # lemm

    trans = WordNetLemmatizer()
    trans_text_lem = [trans.lemmatize(i) for i in alpha_tokens]
    # stem 
    trans = PorterStemmer()
    trans_text_stem = [trans.stem(i) for i in trans_text_lem]
    
    ####################################################################
    ####################################################################
    
    # in english
    if eng_words :
        engl_text = [i for i in trans_text_stem if i in eng_words]
    else : 
        engl_text = trans_text_stem
    
    ####################################################################
    ####################################################################
    
    # manage return type
    if rejoin :
        return " ".join(engl_text)
    return engl_text    


stopwords_en = set(["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"])
stopwords_nltk = set(stopwords.words('english'))
stopwords_punct = set(punctuation)
stopwords_combined = set.union(stopwords_en, stopwords_nltk, stopwords_punct)

def process_text_slide(doc, rejoin=False, min_len_word=3, force_is_alpha=True, eng_words=None, extra_words=None):

    
    # extra_words
    if not extra_words:
        extra_words = []
    
    # lower and strip
    doc = doc.lower().strip()
    print("==================================lower==========================================")
    print(f"nb tokens {len(doc)}")

    
    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    print("====================================tokenize========================================")
    raw_tokens_list = tokenizer.tokenize(doc)
    print("============================================================================")    
    display_tokens_info(raw_tokens_list)
    
    # classics stopwords
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stopwords_combined]
    print("=================================stopwords===========================================")    
    display_tokens_info(cleaned_tokens_list)
    
    ###############################################################
    ###############################################################
    
    # no more len words
    more_than_N = [w for w in cleaned_tokens_list if len(w) >= min_len_word]
    print("==============================more len words==============================================")    
    display_tokens_info(more_than_N)
    
    # only alpha chars
    if force_is_alpha :
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else :
        alpha_tokens = more_than_N
    print("===============================only alpha chars=============================================")        
    display_tokens_info(alpha_tokens)
        
    ####################################################################
    ####################################################################
    
    # lemm

    trans = WordNetLemmatizer()
    trans_text_lem = [trans.lemmatize(i) for i in alpha_tokens]
    print("================================Lemmatizer============================================")    
    display_tokens_info(trans_text_lem)
    # stem 
    trans = PorterStemmer()
    trans_text_stem = [trans.stem(i) for i in trans_text_lem]
    print("=================================Stemmer===========================================")   
    display_tokens_info(trans_text_stem)
    
    ####################################################################
    ####################################################################
    
    # in english
    if eng_words :
        engl_text = [i for i in trans_text_stem if i in eng_words]
    else : 
        engl_text = trans_text_stem
    print("==============================in english==============================================")       
    display_tokens_info(engl_text)
    
    ####################################################################
    ####################################################################
    
    
def clean_text(doc, rejoin=False, list_rare_words=None,  list_more_words=None, extra_words=None):

    
    # list unique words
    if not list_rare_words:
        list_rare_words = []
    
    # list more words
    if not list_more_words:
        list_more_words = []
    
    # extra_words
    if not extra_words:
        extra_words = []
    
    #test = " ".join(doc)
    ###############################################################
    ###############################################################
    
    # no rare tokens
    non_rare_tokens = [w for w in doc if w not in list_rare_words]
    
    
    ###############################################################
    ###############################################################
    
    # no more tokens
    non_more_tokens = [w for w in non_rare_tokens if w not in list_more_words]
        
    ####################################################################
    ####################################################################

    
    # manage return type
    if rejoin :
        return " ".join(non_more_tokens)
    return non_more_tokens    
    
def return_sentences(tokens):
    
    #Create sentences to get clean text as input for vectors
    
    return " ".join([word for word in tokens])    
 

def freqence_words(data_series, nb):
  
        #Compte les occurrences de chaque mot dans data_series
       # et renvoie les nb les plus fréquents avec leur associés.
        
        
    all_words = []

    for word_list in data_series:
        all_words += word_list
        
    freq_dict = nltk.FreqDist(all_words)

    df = pd.DataFrame.from_dict(freq_dict, orient='index').rename(columns={0:"freq"})

    return df.sort_values(by="freq", ascending=False).head(nb)

def plot_freq_dist(data_df, title, long, larg):
    '''
        Displays a bar chart showing the frequency of the modalities
        for each column of data.
        Parameters
        ----------------
        data  : dataframe
                Working data containing exclusively qualitative data
               
        title : string
                The title to give the plot
        long  : int
                The length of the figure for the plot
        larg  : int
                The width of the figure for the plot
        Returns
         '''

    TITLE_SIZE = 20
    TITLE_PAD = 10
    TICK_SIZE = 12
    LABEL_SIZE = 30
    LABEL_PAD = 20

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    data_to_plot = data_df.reset_index().rename(columns={"index":"words"})
    handle_plot_1 = sns.barplot(x="words", y="freq", data=data_to_plot,
                                label="non renseignées", color="blue", alpha=1)

    _, xlabels = plt.xticks()
    _ = handle_plot_1.set_xticklabels(xlabels, size=TICK_SIZE, rotation=45)
    
    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD) #, fontweight="bold"

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD)#, fontweight="bold"

def add_model_score(df_resultats, df: pd.DataFrame = None, model_name: str = 'none', ARI: float = 0, **kwargs):
    #global df_resultats
    if df is None:
        df = df_resultats
    """ajout les resultats d'un model """
    resultats = dict(model=model_name, ARI=ARI)
    resultats = dict(**resultats, **kwargs)
    df = df.append(resultats, ignore_index=True)
    return df



# Fonction pour afficher la répartition des vraies catégories par cluster

def plot_clust_vs_cat(ser_clust, ser_cat, data, figsize=(8,4),
                                  palette='tab10', ylim=(0,250),
                                  bboxtoanchor=None):
    
    # pivot = data.drop(columns=['description','image'])
    pivot = pd.DataFrame()
    pivot['label']=ser_clust
    pivot['category']=ser_cat
    pivot['count']=1
    pivot = pivot.groupby(by=['label','category']).count().unstack().fillna(0)
    pivot.columns=pivot.columns.droplevel()
    
    colors = sns.color_palette(palette, ser_clust.shape[0]).as_hex()
    pivot.plot.bar(width=0.8,stacked=True,legend=True,figsize=figsize,
                   color=colors, ec='k')

    row_data=data.shape[0]

    if ser_clust.nunique() > 15:
        font = 8 
    else : 
        font = 12

    for index, value in enumerate(ser_clust.value_counts().sort_index(ascending=True)):
        percentage = np.around(value/row_data*100,1)   
        plt.text(index-0.25, value+2, str(percentage)+' %',fontsize=font)

    plt.gca().set(ylim=ylim)
    plt.xticks(rotation=0) 

    plt.xlabel('Clusters',fontsize=14)
    plt.ylabel('Nombre de produits', fontsize=14)
    plt.title('Répartition des vraies catégories par cluster',
              fontweight='bold', fontsize=18)

    if bboxtoanchor is not None:
        plt.legend(bbox_to_anchor=bboxtoanchor)
        
    plt.show()    
    
    return pivot

# Affiche la matrice de confusion
def confusion_matrix(y_true, y_pred, title):
    """ xxx
    Args:
        y_true list(str):
        y_pred list(int):
        title (str): 
    Returns:
        -
    """
    # Create a DataFrame with labels and varieties as columns: df
    df = pd.DataFrame({'Labels': y_true, 'Clusters': y_pred})

    # Create crosstab: ct
    ct = pd.crosstab(df['Labels'], df['Clusters'])

    # plot the heatmap for correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(ct.T, 
                 square=True, 
                 annot=True, 
                 annot_kws={"size": 17},
                 fmt='.2f',
                 cmap='Blues',
                 cbar=False,
                 ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=12)
    ax.set_ylabel("clusters", fontsize=15)
    ax.set_xlabel("labels", fontsize=15)

    plt.show()


# Calcul Tsne, détermination des clusters et calcul ARI entre vrais catégorie et n° de clusters
def ARI_fct(l_cat, features, y_cat_num) :
    time1 = time.time()
    num_labels=len(l_cat)
    tsne = manifold.TSNE(n_components=2, perplexity=30, n_iter=2000, 
                                 init='random', learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(features)
    
    # Détermination des clusters à partir des données après Tsne 
    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)
    cls.fit(X_tsne)
    ARI = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_),4)
    time2 = np.round(time.time() - time1,0)
    print("ARI : ", ARI, "time : ", time2)
    
    return ARI, X_tsne, cls.labels_


# visualisation du Tsne selon les vraies catégories et selon les clusters
def TSNE_visu_fct(X_tsne, y_cat_num, labels, ARI, l_cat) :
    fig = plt.figure(figsize=(15,6))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=l_cat, loc="best", title="Categorie")
    plt.title('Représentation des produits par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Représentation des produits par clusters')
    
    plt.show()
    print("ARI : ", ARI)

#BERT
# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
    #                                   and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(desc_text) :
    lw = lower_start_fct(desc_text)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text


# Fonction de préparation des sentences
def bert_inp_fct(sentences, bert_tokenizer, max_length) :
    input_ids=[]
    token_type_ids = []
    attention_mask=[]
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens = True,
                                              max_length = max_length,
                                              padding='max_length',
                                              return_attention_mask = True, 
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")
    
        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0], 
                             bert_inp['token_type_ids'][0], 
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)
    
    return input_ids, token_type_ids, attention_mask, bert_inp_tot
    

# Fonction de création des features
def feature_BERT_fct(model, model_type, sentences, max_length, b_size, mode='HF') :
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:idx+batch_size], 
                                                                               bert_tokenizer, max_length)
        
        if mode=='HF' :    # Bert HuggingFace
            outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode=='TFhub' : # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids" : input_ids, 
                                 "input_mask" : attention_mask, 
                                 "input_type_ids" : token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']
             
        if step ==0 :
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else :
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot,last_hidden_states))
    
    features_bert = np.array(last_hidden_states_tot).mean(axis=1)
    
    time2 = np.round(time.time() - time1,0)
    print("temps traitement : ", time2)
     
    return features_bert, last_hidden_states

def creer_vecteur_moyen_par_mot(data, text_dim, w2v_model):

    vect_moy = np.zeros((text_dim,), dtype='float32')
    num_words = 0.

    for word in data.split():
        if word in w2v_model.wv.vocab:
            vect_moy = np.add(vect_moy, w2v_model[word])
            num_words += 1.

    if num_words != 0.:
        vect_moy = np.divide(vect_moy, num_words)

    return vect_moy

def feature_USE_fct(embed, sentences, b_size):
    batch_size = b_size
    time1 = time.time()

    for step in range(len(sentences)//batch_size):
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step == 0:
            features = feat
        else:
            features = np.concatenate((features, feat))

    time2 = np.round(time.time() - time1, 0)
    print(f'feature_USE_fct, time_taken = {time2} s')
    return features



















































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
