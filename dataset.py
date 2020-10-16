
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from collections import Counter, defaultdict
import pickle
import os

DEBUG = False

def generate_vocabulary(data, save_vocab):
    """
    Creates a vocabulary from data set.

    Args:
        data (str): All data
        save_vocab (str): path of the file where we save vocab
    Returns:
        list(str): list of words in the vocabulary
    """
    all_data = " ".join(data)

    # Creating a frequency count of each word
    dd = defaultdict(int)
    words = [word.lower() for sent in sent_tokenize(all_data)
             for word in word_tokenize(sent)]
    for word in words:
        dd[word] += 1

    # store our vocab
    vocabulary = []

    # If frequency > 5, add to vocab
    for word, freq in dd.items():
        if freq >= 5:
            vocabulary.append(word)

    # saves the voacab to the mentionmed file
    dirname = os.path.dirname(save_vocab)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(save_vocab, 'wb') as f:
        pickle.dump(vocabulary, f)

    return vocabulary


class HANData(Dataset):

    def __init__(
            self,
            df,
            dict_path,
            task,
            cols_to_train=('tweets', 'judgments'),
            max_length_sentences=10,
            max_length_word=20,
            vocabulary_size=10000):

        super(HANData, self).__init__()
        if(DEBUG):
            print("------------------------------------------------------------------")

        # Extracting features from DataFrame
        texts = df['tweets'].values.tolist()
        labels = df['judgements'].values.tolist()
        if(DEBUG):
            print(texts)
        texts = [str(text).lower() for text in texts]

        self.texts = texts
        self.labels = labels

        # Opening / Creating Pickle file at the given path
        if os.path.isfile(dict_path):
            self.dict = pickle.load(open(dict_path, 'rb'))
        else:
            self.dict = generate_vocabulary(texts, dict_path)

        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        
        if(DEBUG):
            print("LABELSSSSSS")
            print(len(set(self.labels)))
            print("LABELSSSSSS")

        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]

        document_encode = []
        for sentences in sent_tokenize(text=text):
            intermed = []
            for word in word_tokenize(text=sentences):
                if word in self.dict:
                    intermed.append(self.dict.index(word))
                else:
                    intermed.append(-1)
            document_encode.append(intermed)

        for sentences in document_encode:
            extended_words = []
            if len(sentences) < self.max_length_word:
                for _ in range(self.max_length_word - len(sentences)):
                    extended_words.append(-1)
            sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = []
            for _ in range(self.max_length_sentences - len(document_encode)):
                intermed = []
                for __ in range(self.max_length_word):
                    intermed.append(-1)
                extended_sentences.append(intermed)
            document_encode.extend(extended_sentences)

        intermediate_doc = []
        for sentences in document_encode:
            intermediate_doc.append(sentences[:self.max_length_word])

        document_encode = intermediate_doc[:self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label
