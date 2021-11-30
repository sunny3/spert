# Функции для работы с корпусом
import numpy as np
import pandas as pd
from collections import namedtuple
import os
import json


import tqdm

import razdel
import nltk


NTSent = namedtuple("NLTKSent", ["start", "stop", "text"])
NTWord = namedtuple("NLTKWord", ["start", "stop", "text"])


def split_text_on_sentences(text, language):
    """
    Разбивка текста на предложения. Выход аналогичен razdel
    Parameters
    ----------
    text: str
        Текст для разбивки на предложения
    language: str
        Язык, если russian - используется разбивка из пакета razdel,
        для всех других случаев NLTK.tokenize.PunktSentenceTokenizer
    Returns
    -------
    res: list
        Список или итератор по предложениям, каждое предложение - кортеж из 3-х элементов:
            1. индекс символа начала предложение;
            2. индекс символа окончания предложения + 1;
            3. сам текст предложения;
    """

    if language == "russian":
        return list(razdel.sentenize(text))
    else:
        sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        res = []
        doc_sentences_spans = sent_tokenizer.span_tokenize(text)
        for sent_spans in doc_sentences_spans:
            sent_named_tuple = NTSent(start=sent_spans[0], stop=sent_spans[1],
                                      text=text[sent_spans[0]:sent_spans[1]])
            res.append(sent_named_tuple)
        return res


def split_text_on_words(text, language):
    """
    Разбивка текста на слова, аналогичино razdel
    Parameters
    ----------
    text: str
        Текст для разбивки
    language: str
        Язык текста, если russian - используется токенизатор razdel, иначе nltk.tokenize.
    Returns
    -------
    res: list
        Список или итератор слов, каждое слово представлено кортежем из 3-х элементов:
            1. индекс символа начала слова;
            2. индекс символа окончания слова + 1;
            3. сам текст слова;
    """

    if language == "russian":
        return list(razdel.tokenize(text))
    else:
        res = []
        word_tokenizer = nltk.tokenize.WordPunctTokenizer()
        for word_span in word_tokenizer.span_tokenize(text):
            word_named_tuple = NTWord(start=word_span[0], stop=word_span[1],
                                      text=text[word_span[0]:word_span[1]])
            res.append(word_named_tuple)
        return res


def split_doc_on_words(doc, language="russian"):
    """
    Parameters
    ----------
    doc: dict
        Документ из JSON Антона
    language: str
        Язык документа, если russian - для разбивки используется пакет razdel, в остальных случаях Punkt из nltk
    """

    res = []
    doc_text = doc["text"]
    for sent_ind, sent in enumerate(split_text_on_sentences(doc_text, language=language)):
        for word_ind, word in enumerate(split_text_on_words(sent.text, language=language)):
            d_word = {"sent_ind": sent_ind, "sent_start": sent.start, "sent_stop": sent.stop,
                      "word_ind_in_sent": word_ind, "word_start": word.start, "word_stop": word.stop, "word": word.text}
            res.append(d_word)
    res = pd.DataFrame(res)
    res["word_start_in_doc"] = res["word_start"] + res["sent_start"]
    res["word_stop_in_doc"] = res["word_stop"] + res["sent_start"]
    return res