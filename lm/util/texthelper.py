from stanfordcorenlp import StanfordCoreNLP
import math
from scipy import spatial
import unicodedata
import os
import string
import re
import numpy

dir_path = os.path.dirname(os.path.realpath(__file__))
# zh_nlp = StanfordCoreNLP(dir_path + '/../../../lib/stanford-corenlp-full-2017-06-09', lang='zh')

"""
################################################
PREPROCESSING
################################################
"""

def get_word_segments_per_language(seq):
    """
    Get word segments 
    args:
        seq: String
    output:
        word_segments: list of String
    """
    cur_lang = -1 # cur_lang = 0 (english), 1 (chinese)
    words = seq.split(" ")
    temp_words = ""
    word_segments = []

    for i in range(len(words)):
        word = words[i]

        if is_contain_chinese_word(word):
            if cur_lang == -1:
                cur_lang = 1
                temp_words = word
            elif cur_lang == 0: # english
                cur_lang = 1
                word_segments.append(temp_words)
                temp_words = word
            else:
                if temp_words != "":
                    temp_words += " "
                temp_words += word
        else:
            if cur_lang == -1:
                cur_lang = 0
                temp_words = word
            elif cur_lang == 1: # chinese
                cur_lang = 0
                word_segments.append(temp_words)
                temp_words = word
            else:
                if temp_words != "":
                    temp_words += " "
                temp_words += word

    word_segments.append(temp_words)

    return word_segments

def get_pos_tag(seq):
    return zh_nlp.pos_tag(seq)

def remove_emojis(seq):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    seq = emoji_pattern.sub(r'', seq).strip()
    return seq

def remove_punctuation(seq):
    # REMOVE CHINESE PUNCTUATION EXCEPT HYPEN / DASH AND FULL STOP
    seq = re.sub("[\s+\\!\/_,$%=^*?:@&^~`(+\"]+|[+！，。？、~@#￥%……&*（）:;：；《）《》“”()»〔〕]+", " ", seq)
    seq = seq.replace("'", " '")
    seq = seq.replace("’", " ’")
    seq = seq.replace("＇", " ＇")
    seq = seq.replace(".", " ")
    seq = seq.replace("?", " ")
    seq = seq.replace(":", " ")
    seq = seq.replace(";", " ")
    seq = seq.replace("]", " ")
    seq = seq.replace("[", " ")
    seq = seq.replace("}", " ")
    seq = seq.replace("{", " ")
    seq = seq.replace("|", " ")
    seq = seq.replace("_", " ")
    seq = seq.replace("(", " ")
    seq = seq.replace(")", " ")
    seq = seq.replace("=", " ")
    return seq

def remove_special_char(seq):
    seq = re.sub("[·．％°℃×→①ぃγ￣σς＝～•＋δ≤∶／⊥＿ñãíå∈△β［］±]+", " ", seq)
    return seq

def remove_space_in_between_words(seq):
    return seq.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").strip().lstrip()

def remove_return(seq):
    return seq.replace("\n", "").replace("\r", "").replace("\t", "")

def preprocess_mixed_language_sentence(seq, retokenize=True):
    seq = seq.lower()
    seq = seq.replace("\u3000", " ")
    seq = seq.replace("[", " [")
    seq = seq.replace("]", "] ")
    seq = seq.replace("#", "")
    seq = seq.replace(",", "")
    seq = seq.replace("*", "")
    seq = seq.replace("\n", "")
    seq = seq.replace("\r", "")
    seq = seq.replace("\t", "")
    seq = seq.replace("~", "")
    seq = re.sub("[\(\[].*?[\)\]]", "", seq) # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = remove_special_char(seq)
    seq = remove_space_in_between_words(seq)
    seq = seq.strip()
    seq = seq.lstrip()
    
    seq = remove_punctuation(seq)
    seq = remove_space_in_between_words(seq)
    seq = seq.strip()
    seq = seq.lstrip()
    
    # Tokenize chinese characters
    if len(seq) <= 1:
        return ""
    else:
        if retokenize:
            words = zh_nlp.word_tokenize(seq)
        else:
            words = seq.split(" ")
        seq = ""
        for i in range(len(words)):
            seq = seq + words[i]
            if i < len(words) - 1:
                seq = seq + " "

        return seq

def is_chinese_char(cc):
    return unicodedata.category(cc) == 'Lo'

def is_contain_chinese_word(seq):
    for i in range(len(seq)):
        if is_chinese_char(seq[i]):
            return True
    return False