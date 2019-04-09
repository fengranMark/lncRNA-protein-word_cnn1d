# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 21:57:14 2018

@author: MFR
"""

from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,matthews_corrcoef
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
import random
import os
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import SimpleRNN, Activation, Dense, Dropout, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Convolution2D, MaxPooling2D, MaxPooling1D, Convolution1D
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import Bidirectional
import nltk  # 用来分词
import collections  # 用来统计词频
import codecs

def seq_combine(f1, f2, f3, f4):
    f_lnc = codecs.open(f1,'r','utf-8')
    lnc = []                #rna列表
    for line in f_lnc:
        line = line.split('\t')
        lnc.append(line[len(line)-2].lower())
    #print(lnc[0])
    
    inter_action = []       #相互作用元组对(蛋白质标号，rna编号)
    f_inter = codecs.open(f2,'r','utf-8')
    for line in f_inter:
        line = line.split('\t')
        inter_action.append((line[1], str(line[2])))
    #print(inter_action[0])
    
    rbp = {}                 #蛋白质字典   标号：序列
    f_rbp = codecs.open(f3,'r','utf-8')
    for line in f_rbp:
        line = line.split('\t')
        rbp[line[0]] = line[1]
    #print(rbp['B6TP60'])
    
    f = codecs.open(f4,'w','utf-8')
    for i in range(len(lnc)):
        for k,v in rbp.items():    #k是标号，v是序列
            if inter_action.count((k,i+1)):
                f.write('1' + '\t' + lnc[i] + protein_trans(v) + '\n')
            else:
                f.write('0' + '\t' + lnc[i] + protein_trans(v) + '\n')
                
    f_lnc.close()
    f_inter.close()
    f_rbp.close()     
    f.close()
    
    return os.path.abspath(f4)

def protein_trans(str1):
    seq = ""
    dic = {'A':'A','C':'C','D':'D','E':'D','F':'F','G':'A','H':'H','I':'F','K':'K','L':'F','M':'M','N':'H','P':'F','Q':'H','R':'K','S':'M','T':'M','V':'A','W':'H','Y':'M'}
    for i in range(len(str1)):
        seq += (dic.get(str1[i]))
    return seq

def create_dataset(file):
    maxlen = 0  # 句子最大长度
    word_freqs = collections.Counter()  # 词频
    num_recs = 0  # 样本数
    with open(file, 'r+', encoding='gb18030', errors='ignore') as f:
        for line in f:
            label, sentence = line.strip().split("\t")
            words=[]
            count=0
            while(count<len(sentence)):
                if(len(sentence[count:count+5])==5):
                    words.append(sentence[count:count + 5])  # ['tact','actg']
                count += 5
    
            #print(words)
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                word_freqs[word] += 1
    
            num_recs += 1
        #print(word_freqs)
    print('max_len ', maxlen)
    print('nb_words ', len(word_freqs))
    
    MAX_FEATURES = 1024
    MAX_SENTENCE_LENGTH = 2056 #2056
    
    # 接下来建立两个 lookup tables，分别是 word2index 和 index2word，用于'词'和数字转换。
    vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
    word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
    # print(word_freqs.most_common(2000))
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    # print(word2index)
    index2word = {v: k for k, v in word2index.items()}
    
    # 下面就是根据 lookup table 把句子转换成数字序列了，并把长度统一到 MAX_SENTENCE_LENGTH，不够的填0，多出的截掉
    X = np.empty(num_recs,dtype=list)
    y = np.zeros(num_recs)
    i=0
    with open(file, 'r+', encoding='gb18030', errors='ignore') as f:
        for line in f:
            label, sentence = line.strip().split("\t")
            words=[]
            count=0
            while(count<len(sentence)):
                if(len(sentence[count:count+5])==5):
                    words.append(sentence[count:count + 5])  # ['tact','actg']
                count += 5
    
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            y[i] = int(label)
            i += 1
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH,padding='post')
    
    index = [i for i in range(len(X))]  
    random.shuffle(index) 
    X = X[index]
    y = y[index]
    
    return X, y, vocab_size, MAX_SENTENCE_LENGTH

def create_model(vocab_size, MAX_SENTENCE_LENGTH):
    EMBEDDING_SIZE = 256 #256
    HIDDEN_LAYER_SIZE = 128 #128
    filters = 16 #16
    kernel_size = 5
    model = Sequential()
    model.add(Embedding(vocab_size,
                        EMBEDDING_SIZE,
                        input_length = MAX_SENTENCE_LENGTH))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1)) #1
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dense(HIDDEN_LAYER_SIZE))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.5)))
    model.add(Dense(1))
    model.add(Dropout(0.1))
    model.add(Activation('sigmoid'))
    model.compile(loss = 'binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def run_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Training ------------')
    # checkpoint
    filepath="c2d_zea_balance_cut..hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    BATCH_SIZE = 32 #32 91.3
    NUM_EPOCHS = 50 #10 90.7 20 91.3 50 91.0
    #history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(X_test, y_test))
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(X_test, y_test), callbacks=callbacks_list, verbose=0)
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


   

    
    