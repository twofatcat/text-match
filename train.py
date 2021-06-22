# load and pre-processing
#-*- coding: UTF-8 -*-
import sys
import os
import json
import random
from functools import reduce

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import math
from nltk.stem import PorterStemmer

np.random.seed(1111)
random.seed(1111)
#-*- coding: UTF-8 -*-
class DataCorpus(object):  # load and processing
    def __init__(self, data_path, vocab_size = 30000, wordvec_path = None, embed_size = 50):
        df = pd.read_csv(data_path, sep ='\t', error_bad_lines=False, warn_bad_lines=False,low_memory=False)
        # new_df = df.drop_duplicates()
        # new_df = new_df.drop_duplicates(subset=['question2'])
        id = df.columns[0]
        qid1 = df.columns[1]
        qid2 = df.columns[2]
        qs1 = df.columns[3]
        qs2 = df.columns[4]
        dump = df.columns[5]
        df[id] = pd.to_numeric(df[id], downcast='integer', errors='coerce')
        df[qid1] = pd.to_numeric(df[qid1], errors='coerce', downcast='integer')
        df[qid2] = pd.to_numeric(df[qid2], errors='coerce', downcast='integer')
        df[qs1] = df[qs1].astype(str)
        df[qs2] = df[qs2].astype(str)
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.drop_duplicates(subset=['question2'])
        data = df.dropna()

        self.data = data
        self.data100 = self.data[self.data[dump]==1.0][:100]
        self.query = self.clean(self.data100[qs1]).tolist()
        # self.query100 = self.data[self.data[dump]==1.0][:100]
        self.question2 = self.clean(self.data[qs2]).tolist()
        self.emb_size = embed_size
        self.vocab_size = vocab_size
        self.get_id()
        self.all_words(self.vocab_size)
        self.load_word2vec(wordvec_path)
    def all_words(self, vocab_size):
        assert len(self.query) == 100
        assert len(self.question2) == len(self.data)
        q1_all_words = []
        q2_all_words = []
        q1text = [['<s>']+nltk.WordPunctTokenizer().tokenize(text)+['</s>'] for text in self.query]
        q2text = [['<s>']+nltk.WordPunctTokenizer().tokenize(text)+['</s>'] for text in self.question2]
        for tokens in q1text:  # utterances
            q1_all_words.extend(tokens)  # -*- coding: UTF-8 -*- 用extend把列表里边可能有的冗余列表除去了，从而只提取出所有值
        self.q1_vocab_count = Counter(q1_all_words).most_common()  # 单词和每个单词个数
        for tokens in q2text:  # utterances
            q2_all_words.extend(tokens)  # 用extend把列表里边可能有的冗余列表除去了，从而只提取出所有值
        self.q2_vocab_count = Counter(q2_all_words).most_common()  # 单词和每个单词个数
        self.q1_vocab_size = len(self.q1_vocab_count)  # 总单词个数
        self.q2_vocab_size = len(self.q2_vocab_count)  # 总单词个数
        discard_wc = np.sum([c for t, c, in self.q2_vocab_count[vocab_size:]])  # 算被遗弃的单词的总词量
        vocab_count = self.q2_vocab_count[0:vocab_size]  # 记录前10000个单词
        self.vocab = [t for t, cnt in vocab_count]
        print(f"Load corpus of question2, total words are {self.q2_vocab_size}")
        print(f'Keeping {vocab_size} remained and the total discard num is {discard_wc}.')
    def get_id(self):
        self.idlist = {id:idx for idx, id in enumerate(self.data['id'])}
    def load_word2vec(self,wordvec_path):
        with open(wordvec_path, "r") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        # clean up lines for memory efficiency
        self.word2vec = None
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.emb_size) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            vec=np.expand_dims(vec, axis=0)
            self.word2vec=np.concatenate((self.word2vec, vec),0) if self.word2vec is not None else vec
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))
    def clean(self, data):
        stop_words = stopwords.words('english')
        def lower(df):
            new_df = df.apply(lambda x: " ".join(x.lower() for x in x.split()))
            return new_df
        def remove_url(df):  # my assignment1 code
            new_df = df.apply(lambda x: re.sub(r"http\S+", "", x))
            return new_df
        def remove_extra_whitespace_tabs(df):  # my assignment1 code
            pattern = r'^\s*|\s\s*'
            data_list = []
            for text in df:
                new_text = re.sub(pattern, ' ', text).strip()
                data_list.append(new_text)
            new_df = pd.Series(data_list)
            return new_df
        def remove_numbers(df):
            pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
            pattern2 = r'^\s*|\s\s*'
            data_list = []
            for text in df:
                new_text = re.sub(pattern, '', text).strip()
                #         print(new_text)
                new_text = re.sub(pattern2, ' ', new_text).strip()
                data_list.append(new_text)
            new_df = pd.Series(data_list)
            return new_df
        # def remove_stopwords(df):
        #     new_df = df.apply(lambda x: " ".join(m for m in x.split() if m not in stop_words))
        #     return new_df
        def punctuation(df):  #  my assignment1 code
            pun_data = df.str.replace('[^\w\s_]', '')
            return pun_data
        # def adding_bod(df):
        #     new_df = df.apply(lambda x: '<s> '+ " ".join(x for x in x.split())+' </s>')
        #     return new_df
        new_q = lower(data)
        new_q = remove_url(new_q)
        new_q = remove_extra_whitespace_tabs(new_q)
        new_q = remove_numbers(new_q)
        # new_q = remove_stopwords(new_q)
        new_q = punctuation(new_q)
        # new_q = adding_bod(new_q)
        return new_q
        # return (id, qid1, qid2), (question1, question2), (is_duplicate)


class Tf_Idf(object):  # text matching
    def __init__(self, query, corpus, id, corpus_word):
        self.id = id  # dup = 1 时候query的 索引id
        rmcorpus = self.remove_stopwords(corpus)
        rmquery = self.remove_stopwords(query)
        self.corpus = self.stem(rmcorpus)
        self.query = self.stem(rmquery)
        self.vocab()
        self.get_table()
        self.text_matching()

    def remove_stopwords(self, df):
        stop_words = stopwords.words('english')
        new_df = [" ".join(m for m in x.split() if m not in stop_words) for x in df]
        return new_df

    def stem(self, df):
        ps = PorterStemmer()
        new_df = [" ".join(ps.stem(word) for word in x.split()) for x in df]
        return new_df

    def vocab(self):
        words = []
        text = [query.split() for query in self.query]
        for query in text:
            words.extend(query)
        self.word = list(set(words))

    def get_table(self):
        doc_len = len(self.corpus)
        inverse_word = {}
        for ids, sentence in enumerate(self.corpus):
            if ids % 10000 == 0:
                print(f'Having processed {ids} sentences.')
            ls_sen = sentence.split()  # 计算query单词 idf，单词在query里, 计算这个单词的tf，
            count = 0
            sen_length = len(ls_sen)
            word_dict = {}
            for idx, word in enumerate(ls_sen):
                if word in self.word:
                    if word not in inverse_word:
                        inverse_word[word] = []
                    if word not in word_dict:
                        word_dict[word] = 1
                    word_dict[word] += 1
            if word_dict != {}:
                for word in word_dict:
                    inverse_word[word].append((ids, word_dict[word] / sen_length))  # tf

        for word in inverse_word:
            frequency = len(inverse_word[word])
            idf = doc_len / frequency
            inverse_word[word] = [(i[0], i[1] * idf) for i in inverse_word[word]]
        self.inverse_word = inverse_word

    def text_matching(self):
        def get_all_id(matchlist):
            all_id = {}
            for ls in matchlist:
                for j in ls:
                    if j[0] not in all_id:
                        all_id[j[0]] = 0
                    all_id[j[0]] += j[1]
            return all_id

        accuracy2 = 0
        accuracy5 = 0
        self.remember2 = []
        self.remember5 = []
        for i in range(len(self.query)):
            #             print('-------')
            #             print('id----',self.id[i])
            quelist = self.query[i].split()
            match_list = []
            #             print('quelist----',quelist)
            for word in quelist:
                if word in self.inverse_word:
                    wordlist = self.inverse_word[word]
                    wordlist.sort(key=lambda x: x[1], reverse=True)  # [(),()...]
                    match_list.append(wordlist[:10])  # [[(),(),()], []...]
                else:
                    continue
            #             print('match list----',match_list)
            match = get_all_id(match_list)
            #             print('match----',match)
            final = [(match[x], x) for x in match]
            final.sort(key=lambda x: x[0], reverse=True)
            topscore = set()
            for k in final:
                topscore.add(k[0])
            ls_topscore = list(topscore)
            ls_topscore.sort(reverse=True)
            ls_top2 = ls_topscore[:2]
            ls_top5 = ls_topscore[:5]
            #             print('final----',final)
            top2 = [m[1] for m in final if m[0] in ls_top2]  # 头两个的ID
            top5 = [m[1] for m in final if m[0] in ls_top5]  # 头五个的ID
            #             print(self.top2)
            #             print(self.top5)
            #             print(self.id[i])
            if self.id[i] in top2:
                accuracy2 += 1
                self.remember2.append((i, top2))
            if self.id[i] in top5:
                accuracy5 += 1
                self.remember5.append((i, top5))
        accuracy2 /= 100
        accuracy5 /= 100
        self.ac2 = accuracy2
        self.ac5 = accuracy5


class Ave_st_embed(object):
    def __init__(self, query, corpus, glove, vocab, id):
        self.id = id
        self.vocab = vocab
        self.glove = glove
        self.corpus = corpus
        self.query = query
        self.cos_similarity()

    def sentence_embed(self, data):
        table = {}
        for i in range(len(self.vocab)):
            table[self.vocab[i]] = self.glove[i]
        self.table = table
        sentence_list = []
        for text in data:
            ave_embed = [0] * 50
            tmp = text.split()
            count = 0
            for word in tmp:
                if word in table:
                    count += 1
                    ave_embed += self.table[word]
            #             print('ave_embed--',ave_embed)
            ave = ave_embed
            #             print('ave--',ave)
            if count != 0:
                ave = [i / count for i in ave]
            else:
                ave = np.random.randn(50) * 0.1
            sentence_list.append(ave)
        return sentence_list

    def cos_similarity(self):
        accuracy2 = 0
        accuracy5 = 0
        self.embedquery = self.sentence_embed(self.query)
        self.embedcorpus = self.sentence_embed(self.corpus)
        top2 = []
        top5 = []
        self.memory2 = []
        self.memory5 = []
        for i in range(len(self.embedquery)):
            if i % 20 == 0:
                print(f'Having processed {i/100} queries.')
            simi = []
            cos_i = np.array(self.embedquery[i])
            cal_i = math.sqrt(np.dot(cos_i, cos_i))
            for j in range(len(self.embedcorpus)):
                if j % 50000 == 0 and i % 20 == 0:
                    print(f'----Having processed {j} question2.')
                cos_j = np.array(self.embedcorpus[j])
                cos = np.dot(cos_i, cos_j) / (cal_i * math.sqrt(np.dot(cos_j, cos_j)))
                simi.append((cos, j))
            simi.sort(key=lambda x: x[0], reverse=True)
            topscore = set()
            for k in simi:
                topscore.add(k[0])
            ls_topscore = list(topscore)
            ls_topscore.sort(reverse=True)
            ls_top2 = ls_topscore[:2]
            ls_top5 = ls_topscore[:5]
            #             print('final----',final)
            top2 = [m[1] for m in simi if m[0] in ls_top2]  # 头两个的ID
            top5 = [m[1] for m in simi if m[0] in ls_top5]  # 头五个的ID

            if self.id[i] in top2:
                accuracy2 += 1
                self.memory2.append((i, top2))
            if self.id[i] in top5:
                accuracy5 += 1
                self.memory5.append((i, top5))
        accuracy2 /= 100
        accuracy5 /= 100
        self.ac2 = accuracy2
        self.ac5 = accuracy5


class Sif(object):
    def __init__(self, query, corpus, id, vocab_c, table):
        self.vocab_c = vocab_c
        self.total_words = np.sum([t[1] for t in self.vocab_c])
        self.id = id
        self.corpus = corpus
        self.query = query
        self.table = table
        self.cos_similarity()

    def sentence_embed(self, data):
        def SVD(metrix):
            u, s, v = np.linalg.svd(metrix,full_matrices=False)
            return u, v

        p_word_counts = {}
        for i in range(len(self.vocab_c)):
            p_word_counts[self.vocab_c[i][0]] = self.vocab_c[i][1] / self.total_words
        self.word_counts = p_word_counts
        #         print(p_word_counts)

        sentence_list = []
        button = False
        for idx,text in enumerate(data):
            if idx%10000 == 0:
                print(f'Process---{idx}')
            tmp = text.split()
            count = 0
            length_tmp = len(tmp)
            sen_embed = []
            for word in tmp:
                if word in self.table:
                    count += 1
                    sen_embed.append((self.table[word], word))  # [[],[],...]

            sum = np.array([0.0] * 50)
            if count != 0:
                #                 print(sen_embed)
                for k in range(len(sen_embed)):
                    #                     print(sen_embed[k][0])
                    #                     print(sen_embed[k][1])
                    word_vec = np.array(sen_embed[k][0])
                    pos = self.word_counts[sen_embed[k][1]]
                    param = 0.003 / (0.003 + pos)
                    sum = sum + param * word_vec / length_tmp
            else:
                sum = np.random.randn(50) * 0.1
            sum = sum.tolist()
            sentence_list.append(sum)
        #         print(len(sentence_list),sentence_list)
        matrix = np.array(sentence_list).T
        #         print(matrix.shape,matrix)
        if button == False:
            U, V = SVD(matrix)
            u = np.array(U).T[0]
            shape = len(u)
            pa = np.matmul(u.reshape(shape, 1), u.reshape(1, shape))
        button = True
        #         print(U.shape)
        new_sentence_list = []
        for s in sentence_list:
            #             print(pa.shape)
            vec_s = np.array(s).T
            #             print(vec_s.shape)
            new_vec = vec_s - np.matmul(pa, vec_s)
            new_sentence_list.append(new_vec.T.tolist())
        return new_sentence_list
    def cos_similarity(self):
        accuracy2 = 0
        accuracy5 = 0
        self.embedquery = self.sentence_embed(self.query)
        self.embedcorpus = self.sentence_embed(self.corpus)
        print('new sentence embeding loaded')
        self.memory2 = []
        self.memory5 = []
        for i in range(len(self.embedquery)):
            if i % 20 == 0:
                print(f'Having processed {i/100} queries.')
            simi = []
            cos_i = np.array(self.embedquery[i])
            cal_i = math.sqrt(np.dot(cos_i, cos_i))
            for j in range(len(self.embedcorpus)):
                if j % 50000 == 0 and i % 20 == 0:
                    print(f'----Having processed {j} question2.')
                cos_j = np.array(self.embedcorpus[j])
                cos = np.dot(cos_i, cos_j) / (cal_i * math.sqrt(np.dot(cos_j, cos_j)))
                simi.append((cos, j))
            simi.sort(key=lambda x: x[0], reverse=True)
            topscore = set()
            for k in simi:
                topscore.add(k[0])
            ls_topscore = list(topscore)
            ls_topscore.sort(reverse=True)
            ls_top2 = ls_topscore[:2]
            ls_top5 = ls_topscore[:5]
            #             print('final----',final)
            top2 = [m[1] for m in simi if m[0] in ls_top2]  # 头两个的ID
            top5 = [m[1] for m in simi if m[0] in ls_top5]  # 头五个的ID

            if self.id[i] in top2:
                accuracy2 += 1
                self.memory2.append((i, top2))
            if self.id[i] in top5:
                accuracy5 += 1
                self.memory5.append((i, top5))
        accuracy2 /= 100
        accuracy5 /= 100
        self.ac2 = accuracy2
        self.ac5 = accuracy5

data_path = os.path.abspath(".")+'/data.tsv'
wordvec_path = os.path.abspath(".")+'/glove.twitter.27B.50d.txt'
data = DataCorpus(data_path, wordvec_path=wordvec_path)

id = data.data100.index.tolist()
print('TF-IDF\n')
tf_matching = Tf_Idf(data.query,data.question2,id,data.q2_vocab_count)
print('Average embedding\n')
ave = Ave_st_embed(data.query,data.question2,data.word2vec,data.vocab,id)
print('SIF embedding\n')
sif = Sif(data.query,data.question2,id,data.q2_vocab_count,ave.table)
# corpus = DataCorpus()
# train_set, test_set, valid_set = corpus["train"], corpus["test"], corpus["valid"]
# query = sys.argv[1]

with open('output.txt', 'w') as file:
    file.write('TFIDF accuracy:')
    file.write('\n-----')
    file.write(str(tf_matching.ac2) + ' '+ str(tf_matching.ac5))
    file.write('\n')
    file.write('average embedding accuracy:')
    file.write('\n')
    file.write(str(ave.ac2) + ' '+ str(ave.ac5))
    file.write('\n')
    file.write('sif accuracy:')
    file.write('\n')
    file.write(str(sif.ac2) + ' '+ str(sif.ac5))

tfidf=tf_matching.remember5
average=ave.memory5
sif=sif.memory5

# class AnswerSearching(object):
#     def __init__(self, query, datalist=data.query, idlist=id, tfidf=tfidf,\
#                  average=average, sif=sif.sif, data=data.question2):
#         self.sif = sif
#         self.average = average
#         self.tfidf = tfidf
#         self.query = query
#         self.data = data
#         for idx, question1 in enumerate(datalist):
#             if question1 == self.query:
#                 self.id = idlist[idx]
#                 break
#         for i in self.tfidf:
#             if i[0] == self.id:
#                 print(f'Top 5 TF-IDF id: {i[1]}')
#                 print(f'{[self.data[id] for id in i[1]]}')
#                 break
#         for i in self.average:
#             if i[0] == self.id:
#                 print(f'Top 5 average embedding id: {i[1]}')
#                 print(f'{[self.data[id] for id in i[1]]}')
#                 break
#         for i in self.sif:
#             if i[0] == self.id:
#                 print(f'Top 5 SIF embedding id: {i[1]}')
#                 print(f'{[self.data[id] for id in i[1]]}')
#                 break
def clean(data):

    def lower(df):
        new_df = " ".join(x.lower() for x in df.split())
        return new_df

    def remove_url(df):  # my assignment1 code
        new_df = re.sub(r"http\S+", "", df)
        return new_df

    def remove_extra_whitespace_tabs(df):  # my assignment1 code
        pattern = r'^\s*|\s\s*'
        data_list = []
        for text in df.split():
            new_text = re.sub(pattern, ' ', text).strip()
            data_list.append(new_text)
        new_df = " ".join(x for x in data_list)
        return new_df

    def remove_numbers(df):
        pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
        pattern2 = r'^\s*|\s\s*'
        data_list = []
        for text in df.split():
            new_text = re.sub(pattern, '', text).strip()
            #         print(new_text)
            new_text = re.sub(pattern2, ' ', new_text).strip()
            data_list.append(new_text)
        new_df = " ".join(x for x in data_list)
        return new_df

    # def remove_stopwords(df):
    #     new_df = df.apply(lambda x: " ".join(m for m in x.split() if m not in stop_words))
    #     return new_df
    def punctuation(df):  # my assignment1 code
        pun_data = re.sub('[^\w\s_]', '',df)
        return pun_data

    # def adding_bod(df):
    #     new_df = df.apply(lambda x: '<s> '+ " ".join(x for x in x.split())+' </s>')
    #     return new_df
    new_q = lower(data)
    new_q = remove_url(new_q)
    new_q = remove_extra_whitespace_tabs(new_q)
    new_q = remove_numbers(new_q)
    # new_q = remove_stopwords(new_q)
    new_q = punctuation(new_q)
    # new_q = adding_bod(new_q)
    return new_q

if __name__ == '__main__':
    query = clean(sys.argv[1])
    for idx, question1 in enumerate(data.query):
        if question1 == query:
            idname = id[idx]
            break
    for i in tfidf:
        if i[0] == idname:
            print(f'Top 5 TF-IDF id: {i[1]}')
            print(f'{[self.data[id] for id in i[1]]}')
            break
    for i in average:
        if i[0] == idname:
            print(f'Top 5 average embedding id: {i[1]}')
            print(f'{[self.data[id] for id in i[1]]}')
            break
    for i in sif:
        if i[0] == idname:
            print(f'Top 5 SIF embedding id: {i[1]}')
            print(f'{[self.data[id] for id in i[1]]}')
            break
    # print(sys.argv[1]+'yumin')
