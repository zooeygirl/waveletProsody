import os
import sys
import random
import math
from torch.utils import data
from pytorch_transformers import BertTokenizer, GPT2Tokenizer
import torch
import numpy as np
from transformers import AutoTokenizer

class Dataset(data.Dataset):
    #def __init__(self, tagged_sents, tag_to_index, config, word_to_embid=None):
    def __init__(self, tagged_sents, file_ids, tag_to_index, config, word_to_embid=None):
        sents, tags_li,values_li, prevSents, prevTags, prevValues = [], [], [], [], [], [] # list of lists
        self.config = config

        for j, sent in enumerate(tagged_sents):
            book = file_ids[j][0]
            spk = file_ids[j][2]
            prevWord = []
            prevTag = []
            prevVal = []
            if book in ["Emma",'SenseAndSensibility', 'PrideAndPrejudice', 'Persuasion', 'MansfieldPark', 'JaneEyre']:
              ch = file_ids[j][1].split('-')[0]
              numPrev = 2
              for k in range(1, numPrev):
                ut = str(int(file_ids[j][1].split('-')[1])-k)
                if not int(ut) < 0:
                  if book =='PrideAndPrejudice':
                    if ch+'-'+ut =='001-0' or ch+'-'+ut =='001-1' or ch+'-'+ut =='001-2':
                      print(ch+'-'+ut+" "+ spk)
                      prev = file_ids.index((book, ch+'-'+ut, spk ))
                      print([word_tag[0] for word_tag in tagged_sents[prev]])
                  try:
                      prev = file_ids.index((book, ch+'-'+ut, spk))
                      prevWord = [word_tag[0] for word_tag in tagged_sents[prev]] + prevWord
                      prevTag = [word_tag[1] for word_tag in tagged_sents[prev]] + prevTag
                      prevVal = [word_tag[3] for word_tag in tagged_sents[prev]] + prevVal
                  except:
                      print((book, ch+'-'+ut, spk))

            prevSents.append(prevWord)
            prevTags.append(prevTag)
            prevValues.append(prevVal)

            words = prevWord + [word_tag[0] for word_tag in sent]
            tags = prevTag + [word_tag[1] for word_tag in sent]
            values = prevVal + [word_tag[3] for word_tag in sent] #+++HANDE

            #words = [word_tag[0] for word_tag in sent]
            #tags =  [word_tag[1] for word_tag in sent]
            #values = [word_tag[3] for word_tag in sent]

            if self.config.model != 'LSTM' and self.config.model != 'BiLSTM' and config.gpt == 0:
                sents.append(["[CLS]"] + words + ["[SEP]"])
                tags_li.append(["<pad>"] + tags + ["<pad>"])
                values_li.append(["<pad>"] + values + ["<pad>"])
            else:
                sents.append(words)
                tags_li.append(tags)
                values_li.append(values)


        self.sents, self.tags_li, self.values_li, self.prevSents, self.prevTags, self.prevValues, self.file_ids = sents, tags_li, values_li, prevSents, prevTags, prevValues, file_ids
        if self.config.model == 'BertUncased' or self.config.model == 'Transformer':
            if config.gpt != 0:
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            else:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
            #self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        self.tag_to_index = tag_to_index
        self.word_to_embid = word_to_embid

    def __len__(self):
        return len(self.sents)

    def convert_tokens_to_emb_ids(self, tokens):
        UNK_id = self.word_to_embid.get('UNK')
        return [self.word_to_embid.get(token, UNK_id) for token in tokens]

    def __getitem__(self, id):
        words, tags, values_li, prevSents, prevTags, prevValues, file_id = self.sents[id], self.tags_li[id], self.values_li[id], self.prevSents[id], self.prevTags[id], self.prevValues[id], self.file_ids[id] # words, tags, values: string list

        x, y, values, px, pt, pv = [], [], [], [], [], [] # list of ids
        is_main_piece = [] # only score the main piece of each word
        is_main_piece_prev = []
        for w, t, v in zip(words, tags, values_li):
            if self.config.model in ['LSTM', 'BiLSTM', 'LSTMRegression']:
                tokens = [w]
                xx = self.convert_tokens_to_emb_ids(tokens)
            else:
                if self.config.gpt != 0:
                    if t!='NA' and w!="n't" and w!="'d" and w!="'":
                        tokens = self.tokenizer.tokenize(" "+w) if w not in ("[CLS]", "[SEP]") else [w]
                        xx = self.tokenizer.convert_tokens_to_ids(tokens)
                    else:
                        tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                        xx = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                    xx = self.tokenizer.convert_tokens_to_ids(tokens)

            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.tag_to_index[each] for each in t]  # (T,)

            head = [1] + [0]*(len(tokens) - 1) # identify the main piece of each word

            x.extend(xx)
            is_main_piece.extend(head)
            y.extend(yy)
        for p, yt in zip(prevSents, prevTags):
          ptokens = self.tokenizer.tokenize(p) if p not in ("[CLS]", "[SEP]") else [p]
          pxx = self.tokenizer.convert_tokens_to_ids(ptokens)
          px.extend(pxx)
          yt = [yt] + ["<pad>"] * (len(ptokens) - 1)
          yyt = [self.tag_to_index[each] for each in yt]
          pt.extend(yyt)
          head = [1] + [0]*(len(ptokens) - 1) # identify the main piece of each word
          is_main_piece_prev.extend(head)

        #print(is_main_piece)
        #is_main_piece[:len(prevSents)] = [0 for num in range(1,len(prevSents)+1)]
        #print(is_main_piece)


        assert len(x) == len(y) == len(is_main_piece), "len(x)={}, len(y)={}, len(is_main_piece)={}".format(len(x), len(y), len(is_main_piece))
        # seqlen
        seqlen = len(y)
        prvSeqLen = len(px)
        #print('sl',seqlen)
        #print('psl',prvSeqLen)
        #print('pt', len(pt))

        # to string
        words = " ".join(words)
        #print(words)
        prevWords = " ".join(prevSents)
        #print(prevWords)
        tags = " ".join(tags)

        if self.config.log_values:
            # Use log-values to remove affects of 0-skewed value distribution
            values = [np.log(np.log(float(v) + 1)+1) if v not in ['<pad>','NA', 'NA\n'] else self.config.invalid_set_to for v in values_li]
        else:
            values = [float(v) if v not in ['<pad>', 'NA', 'NA\n'] else self.config.invalid_set_to for v in values_li]
            pv = [float(v) if v not in ['<pad>', 'NA', 'NA\n'] else self.config.invalid_set_to for v in prevValues]


        return words, x, is_main_piece, tags, y, seqlen, values, self.config.invalid_set_to, px, prvSeqLen, pt, pv, is_main_piece_prev, prevWords, file_id


def load_dataset(config):
    splits = dict()
    fileDict = dict()
    words = []
    all_sents = []
    for split in ['train', 'dev', 'test']:
        tagged_sents = []
        file_ids = []
        if split == 'train':
            filename = config.train_set
        elif split == 'test':
            filename = config.test_set
        else:
            filename = split
        with open(config.datadir+'/'+filename+'.txt') as f:
            lines = f.readlines()
            if config.fraction_of_train_data < 1 and split == 'train':
                slice = len(lines) * config.fraction_of_train_data
                lines = lines[0:int(round(slice))]
            sent = []
            for i, line in enumerate(lines):
                line = line.replace('\n','')
                split_line = line.split('\t')
                if i != 0 and split_line[0] != "<file>" and i+1 !=len(lines):
                    word = split_line[0]
                    tag_prominence = split_line[1]
                    tag_boundary = split_line[2]
                    value_prominance = split_line[3]
                    value_boundary = split_line[4]

                    # Modify tag value if we specified a different config.nclasses
                    # than default value of 3
                    if config.nclasses == 2:
                        if tag_prominence == '2': tag_prominence = '1' #Collapse the non-0 classes
                    elif config.nclasses > 3:
                        tag_prominence = rediscretize_tag(value_prominance, config.nclasses)

                    sent.append((word, tag_prominence, tag_boundary, value_prominance, value_boundary))
                    words.append(word)
                #elif (i != 0 and split_line[0] == "<file>") or i+1 == len(lines):
                elif (split_line[0] == "<file>") or i+1 == len(lines):
                    if i != 0:
                      tagged_sents.append(sent)
                      sent = []
                    if i+1 != len(lines):
                      utt = split_line[1].split('/')[-1][:-4]
                      try:
                        book = split_line[1].split('/')[-4]
                        spk = split_line[1].split('/')[-3]
                      except:
                        book = split_line[1]
                        spk = 'speaker'
                      file_ids.append((book, utt, spk))



        config.shuffle_sentences = True
        if config.shuffle_sentences:
            #print('do not shuffle')
            c = list(zip(tagged_sents, file_ids))
            random.shuffle(c)
            tagged_sents, file_ids = zip(*c)
            tagged_sents = list(tagged_sents)
            file_ids = list(file_ids)
            #random.shuffle(tagged_sents)

        splits[split] = tagged_sents
        fileDict[split] = file_ids
        all_sents = all_sents + tagged_sents

    vocab = []
    for token in words:
        if token not in vocab:
            vocab.append(token)
    vocab = set(vocab)

    tags = list(set(word_tag[1] for sent in all_sents for word_tag in sent))
    tags = ["<pad>"] + tags

    tag_to_index = {tag: index for index, tag in enumerate(tags)}
    index_to_tag = {index: tag for index, tag in enumerate(tags)}

    print('Training sentences: {}'.format(len(splits["train"])))
    print('Dev sentences: {}'.format(len(splits["dev"])))
    print('Test sentences: {}'.format(len(splits["pAndp"])))

    if config.sorted_batches:
        random.shuffle(splits["train"])
        splits["train"].sort(key=len)

    return splits, tag_to_index, index_to_tag, vocab, fileDict


def pad(batch):
    # Pad sentences to the longest sample
    f = lambda x: [sample[x] for sample in batch]
    #words = f(0)
    words = f(0)
    is_main_piece = f(2)
    tags = f(3)
    seqlens = f(5)
    prevSeq = f(8)
    prvSeqLen = f(9)
    prevWords = f(13)
    file_id = f(14)
    #print('prevlen', prvSeqLen)
    pTags = f(10)
    #print("ptags", [len(p) for p in pTags])
    #print("pvalues", [len(p) for p in f(11)])
    maxlen = np.array(seqlens).max()
    #maxlen = (np.array(seqlens) + np.array(prvSeqLen)).max()
    #print('ml', maxlen)
    invalid_set_to = f(7)[0]

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    #f = lambda x, seqlen: [sample[8] + sample[x] + [0] * (seqlen - len(sample[x])-sample[9]) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    #print("x",x[0])
    #f = lambda x, seqlen: [sample[10] + sample[x]+ [0] * (seqlen - len(sample[x])-sample[9]) for sample in batch] # 0: <pad>
    y = f(4, maxlen)
    #print("y",y[0])

    f = lambda x, seqlen: [sample[x] + [invalid_set_to] * (seqlen - len(sample[x])) for sample in batch] #invalid values are NA and <pad>
    #f = lambda x, seqlen: [sample[12] + sample[x] + [invalid_set_to] * (seqlen - len(sample[x])-sample[9]) for sample in batch] #invalid values are NA and <pad>
    values = f(6, maxlen)

    f = torch.LongTensor
    return words, f(x), is_main_piece, tags, f(y), seqlens, torch.FloatTensor(values), invalid_set_to , prevSeq, prvSeqLen, prevWords, file_id




def load_embeddings(config, vocab):
    vocab.add('UNK')
    word2id = {word: id for id, word in enumerate(vocab)}
    embed_size = 300
    vocab_size = len(vocab)
    sd = 1 / np.sqrt(embed_size)
    weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    weights = weights.astype(np.float32)
    with open(config.embedding_file, encoding='utf8', mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]

            # If word is in our vocab, then update the corresponding weights
            id = word2id.get(word, None)
            if id is not None and len(line) == 301:
                weights[id] = np.array([float(val) for val in line[1:]])

    return weights, word2id

def rediscretize_tag(value_prominance, nclasses):
    if value_prominance == 'NA':
        return 'NA'

    # Simple dividing into bins:
    SOFT_MAX_BOUND = 6.0
    return str(int(min(float(value_prominance) * nclasses / SOFT_MAX_BOUND, nclasses)))
