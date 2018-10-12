# -*- coding: utf-8 -*-
"""
TF-IDF Computation.

AUTHOR: Yue Peng
EMAIL: ypeng7@outlook.com
DATE: 2018.10.03
"""
import os, sys, math
import numpy as np
from sklearn.decomposition import TruncatedSVD
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))


tokenize = lambda doc: doc.strip().lower().split(" ")


def term_frequency(term, tokenized_document):
    """Computed TF
    
    Use count method of list in Python.
    
    Arguments:
        term {str} -- [description]
        tokenized_document {list<str>} -- [description]
    
    Returns:
        [int] -- [description]
    """
    return tokenized_document.count(term)


def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)


def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))


def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values


def tfidf(documents):
    global tokenize
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

def word_tfidf(documents):
    """Computed the tfidf score for each word
    
    TODO: weight equals zero
    
    Arguments:
        documents {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    for document in tokenized_documents:
        word_score = {}
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            word_score[term] = tf * idf[term]
    return word_score


def word_idf(documents):
    """Computed the idf score for each word
    
    [description]
    
    Arguments:
        documents {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return inverse_document_frequencies([tokenize(d) for d in documents])


def compute_pc(X, npc=1):
    """Remove the projection on the principal components
    
    [description]
    
    Arguments:
        X {[type]} -- [description]
    
    Keyword Arguments:
        npc {number} -- [description] (default: {1})
    
    Returns:
        [type] -- [description]
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def get_weighted_average(wv, idxs, w):
    """
    Compute the weighted average vectors
    :param wv: wv[i,:] is the vector for word i
    :param idxs: idxs[i] are the indices of the words in sentence i
    :param w: w[i] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = len(idxs)
    emb = np.zeros([n_samples, wv.shape[1]])
    for i in range(n_samples):
        cnt_zero = np.count_nonzero(w[i])
        # deal with situation that there is no score for all the words
        if cnt_zero == 0:
            cnt_zero += 1
        emb[i, :] = w[i].dot(wv[idxs[i], :]) / cnt_zero
    return emb


def sif_embedding(emb):
    """SIF embedding
    
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    
    Arguments:
        emb {np.array} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return remove_pc(emb)
