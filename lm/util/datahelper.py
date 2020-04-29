import os
import string
import re
import numpy
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
import math
from scipy import spatial
import unicodedata
from scipy.spatial.distance import cosine

from . import texthelper

dir_path = os.path.dirname(os.path.realpath(__file__))

def read_seame_phase1():
    """
    Recursively iterate phase 1 directories and read all the data

    """
    print("> read SEAME corpus")
    interview_phase1_dir = dir_path + "/../../../dataset/seame_LDC2015S04/seame/data/interview/transcript/phaseI/"
    conversation_phase1_dir = dir_path + "/../../../dataset/seame_LDC2015S04/seame/data/conversation/transcript/phaseI/"

    interview_phase1_filenames = []
    conversation_phase1_filenames = []

    interview_phase1_data = {}
    conversation_phase1_data = {}
    
    all_data = {}
    vocab = {}
    speaker_ids = {}

    for root, dirs, files in os.walk(interview_phase1_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                interview_phase1_filenames.append(path)

    for root, dirs, files in os.walk(conversation_phase1_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                conversation_phase1_filenames.append(path)
                
    print("################################")
    print("             SUMMARY            ")
    print("################################")
    print("interview phase 1 files\t\t:", len(interview_phase1_filenames))
    print("conversation phase 1 files\t:", len(conversation_phase1_filenames))
    print("################################\n")

    total_utterances_interview_phase1 = 0
    total_utterances_conversation_phase1 = 0

    total_utterances_interview_phase1_filtered = 0
    total_utterances_conversation_phase1_filtered = 0

    print("> read interview phase 1")
    for i in tqdm(range(len(interview_phase1_filenames))):
        filename = interview_phase1_filenames[i]
        with open(filename, "r") as file:
            for line in file:
                str_id = line.split("_")[0]
                speaker_id = str_id[0:4]
                speaker_ids[speaker_id] = True
                
                arr = line.split("\t")
                seq = arr[3]
                
                seq = texthelper.preprocess_mixed_language_sentence(seq)
                total_utterances_interview_phase1 += 1
                
                if seq != "":
                    total_utterances_interview_phase1_filtered += 1
                    words = seq.split(" ")
                    
                    for j in range(len(words)):
                        vocab[words[j]] = True

                    if speaker_id in interview_phase1_data:
                        interview_phase1_data[speaker_id].append(seq)
                        all_data[speaker_id].append(seq)
                    else:
                        interview_phase1_data[speaker_id] = []
                        interview_phase1_data[speaker_id].append(seq)
                        all_data[speaker_id] = []
                        all_data[speaker_id].append(seq)
                        
    print("> read conversation phase 1")
    for i in tqdm(range(len(conversation_phase1_filenames))):
        filename = conversation_phase1_filenames[i]
        with open(filename, "r") as file:
            for line in file:
                str_id = line.split("_")[0]
                speaker_id = str_id[2:6]
                speaker_ids[speaker_id] = True
                
                arr = line.split("\t")
                seq = arr[3]
                
                seq = texthelper.preprocess_mixed_language_sentence(seq)
                total_utterances_conversation_phase1 += 1

                if seq != "":
                    total_utterances_conversation_phase1_filtered += 1
                    words = seq.split(" ")
                    
                    for j in range(len(words)):
                        vocab[words[j]] = True

                    if speaker_id in conversation_phase1_data:
                        conversation_phase1_data[speaker_id].append(seq)
                        all_data[speaker_id].append(seq)
                    else:
                        conversation_phase1_data[speaker_id] = []
                        conversation_phase1_data[speaker_id].append(seq)
                        all_data[speaker_id] = []
                        all_data[speaker_id].append(seq)
    
    total_utterances = 0
    for key in all_data:
        total_utterances += len(all_data[key])
        
    print("################################")
    print("            OVERVIEW            ")
    print("################################")
    print("number of speaker by speaker_ids:", len(speaker_ids))
    print("number of speaker of all utterances:", len(all_data))
    print("all utterances:", total_utterances)
    print("number of words", len(vocab))
    print("total utterances interview_phase1\t:", total_utterances_interview_phase1)
    print("total utterances conversation_phase1\t:", total_utterances_conversation_phase1)
    
    print("total utterances interview_phase1_filtered\t:", total_utterances_interview_phase1_filtered)
    print("total utterances conversation_phase1_filtered\t:", total_utterances_conversation_phase1_filtered)
    print("################################")
    
    return interview_phase1_data, conversation_phase1_data, all_data, vocab

def read_seame():
    """
    Recursively iterate directories and read all the data

    """
    print("> read SEAME corpus")
    interview_phase1_dir = dir_path + "/../../../dataset/seame_LDC2015S04/seame/data/interview/transcript/phaseI/"
    interview_phase2_dir = dir_path + "/../../../dataset/seame_LDC2015S04/seame/data/interview/transcript/phaseII/"
    conversation_phase1_dir = dir_path + "/../../../dataset/seame_LDC2015S04/seame/data/conversation/transcript/phaseI/"
    conversation_phase2_dir = dir_path + "/../../../dataset/seame_LDC2015S04/seame/data/conversation/transcript/phaseII/"

    interview_phase1_filenames = []
    interview_phase2_filenames = []
    conversation_phase1_filenames = []
    conversation_phase2_filenames = []

    interview_phase1_data = {}
    interview_phase2_data = {}
    conversation_phase1_data = {}
    conversation_phase2_data = {}
    
    all_data = {}
    vocab = {}
    speaker_ids = {}

    for root, dirs, files in os.walk(interview_phase1_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                interview_phase1_filenames.append(path)
                
    for root, dirs, files in os.walk(interview_phase2_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                interview_phase2_filenames.append(path)

    for root, dirs, files in os.walk(conversation_phase1_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                conversation_phase1_filenames.append(path)

    for root, dirs, files in os.walk(conversation_phase2_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                conversation_phase2_filenames.append(path)
                
    print("################################")
    print("             SUMMARY            ")
    print("################################")
    print("interview phase 1 files\t\t:", len(interview_phase1_filenames))
    print("interview phase 2 files\t\t:", len(interview_phase2_filenames))
    print("conversation phase 1 files\t:", len(conversation_phase1_filenames))
    print("conversation phase 2 files\t:", len(conversation_phase2_filenames))
    print("################################\n")

    total_utterances_interview_phase1 = 0
    total_utterances_interview_phase2 = 0
    total_utterances_conversation_phase1 = 0
    total_utterances_conversation_phase2 = 0

    total_utterances_interview_phase1_filtered = 0
    total_utterances_interview_phase2_filtered = 0
    total_utterances_conversation_phase1_filtered = 0
    total_utterances_conversation_phase2_filtered = 0

    print("> read interview phase 1")
    for i in tqdm(range(len(interview_phase1_filenames))):
        filename = interview_phase1_filenames[i]
        with open(filename, "r") as file:
            for line in file:
                str_id = line.split("_")[0]
                speaker_id = str_id[0:4]
                speaker_ids[speaker_id] = True
                
                arr = line.split("\t")
                seq = arr[3]
                
                seq = texthelper.preprocess_mixed_language_sentence(seq)
                total_utterances_interview_phase1 += 1
                
                if seq != "":
                    total_utterances_interview_phase1_filtered += 1
                    words = seq.split(" ")
                    
                    for j in range(len(words)):
                        vocab[words[j]] = True

                    if speaker_id in interview_phase1_data:
                        interview_phase1_data[speaker_id].append(seq)
                        all_data[speaker_id].append(seq)
                    else:
                        interview_phase1_data[speaker_id] = []
                        interview_phase1_data[speaker_id].append(seq)
                        all_data[speaker_id] = []
                        all_data[speaker_id].append(seq)
                        
    print("> read interview phase 2")
    for i in tqdm(range(len(interview_phase2_filenames))):
        filename = interview_phase2_filenames[i]
        with open(filename, "r") as file:
            for line in file:
                str_id = line.split("_")[0]
                speaker_id = str_id[0:4]
                speaker_ids[speaker_id] = True
                
                arr = line.split("\t")
                seq = arr[4]
                
                seq = texthelper.preprocess_mixed_language_sentence(seq, retokenize=False)
                total_utterances_interview_phase2 += 1

                if seq != "":
                    total_utterances_interview_phase2_filtered += 1
                    words = seq.split(" ")
                    
                    for j in range(len(words)):
                        vocab[words[j]] = True

                    if speaker_id in interview_phase2_data:
                        interview_phase2_data[speaker_id].append(seq)
                        all_data[speaker_id].append(seq)
                    else:
                        interview_phase2_data[speaker_id] = []
                        interview_phase2_data[speaker_id].append(seq)
                        all_data[speaker_id] = []
                        all_data[speaker_id].append(seq)
                        
    print("> read conversation phase 1")
    for i in tqdm(range(len(conversation_phase1_filenames))):
        filename = conversation_phase1_filenames[i]
        with open(filename, "r") as file:
            for line in file:
                str_id = line.split("_")[0]
                speaker_id = str_id[2:6]
                speaker_ids[speaker_id] = True
                
                arr = line.split("\t")
                seq = arr[3]
                
                seq = texthelper.preprocess_mixed_language_sentence(seq)
                total_utterances_conversation_phase1 += 1

                if seq != "":
                    total_utterances_conversation_phase1_filtered += 1
                    words = seq.split(" ")
                    
                    for j in range(len(words)):
                        vocab[words[j]] = True

                    if speaker_id in conversation_phase1_data:
                        conversation_phase1_data[speaker_id].append(seq)
                        all_data[speaker_id].append(seq)
                    else:
                        conversation_phase1_data[speaker_id] = []
                        conversation_phase1_data[speaker_id].append(seq)
                        all_data[speaker_id] = []
                        all_data[speaker_id].append(seq)
                        
    print("> read conversation phase 2")
    for i in tqdm(range(len(conversation_phase2_filenames))):
        filename = conversation_phase2_filenames[i]
        with open(filename, "r") as file:
            for line in file:
                str_id = line.split("_")[0]
                speaker_id = str_id[2:6]
                speaker_ids[speaker_id] = True
                
                arr = line.split("\t")
                seq = arr[4]

                seq = texthelper.preprocess_mixed_language_sentence(seq, retokenize=False)
                total_utterances_conversation_phase2 += 1
                
                if seq != "":
                    total_utterances_conversation_phase2_filtered += 1
                    words = seq.split(" ")
                    
                    for j in range(len(words)):
                        vocab[words[j]] = True

                    if speaker_id in conversation_phase2_data:
                        conversation_phase2_data[speaker_id].append(seq)
                        all_data[speaker_id].append(seq)
                    else:
                        conversation_phase2_data[speaker_id] = []
                        conversation_phase2_data[speaker_id].append(seq)
                        all_data[speaker_id] = []
                        all_data[speaker_id].append(seq)
    
    total_utterances = 0
    for key in all_data:
        total_utterances += len(all_data[key])
        
    print("################################")
    print("            OVERVIEW            ")
    print("################################")
    print("number of speaker by speaker_ids:", len(speaker_ids))
    print("number of speaker of all utterances:", len(all_data))
    print("all utterances:", total_utterances)
    print("number of words", len(vocab))
    print("total utterances interview_phase1\t:", total_utterances_interview_phase1)
    print("total utterances interview_phase2\t:", total_utterances_interview_phase2)
    print("total utterances conversation_phase1\t:", total_utterances_conversation_phase1)
    print("total utterances conversation_phase2\t:", total_utterances_conversation_phase2)

    print("total utterances interview_phase1_filtered\t:", total_utterances_interview_phase1_filtered)
    print("total utterances interview_phase2_filtered\t:", total_utterances_interview_phase2_filtered)
    print("total utterances conversation_phase1_filtered\t:", total_utterances_conversation_phase1_filtered)
    print("total utterances conversation_phase2_filtered\t:", total_utterances_conversation_phase2_filtered)
    print("################################")
    
    return interview_phase1_data, interview_phase2_data, conversation_phase1_data, conversation_phase2_data, all_data, vocab

def load_seame_numpy_array():
    interview_phase1_data = numpy.load(dir_path + "/../data/seame/numpy_array/interview_phase1_data.npy", encoding="latin1")
    interview_phase2_data = numpy.load(dir_path + "/../data/seame/numpy_array/interview_phase2_data.npy", encoding="latin1")
    conversation_phase1_data = numpy.load(dir_path + "/../data/seame/numpy_array/conversation_phase1_data.npy", encoding="latin1")
    conversation_phase2_data = numpy.load(dir_path + "/../data/seame/numpy_array/conversation_phase2_data.npy", encoding="latin1")
    vocab = numpy.load(dir_path + "/../data/seame/numpy_array/vocab.npy", encoding="latin1")

    return interview_phase1_data, interview_phase2_data, conversation_phase1_data, conversation_phase2_data, vocab

def save_seame(interview_phase1_data, conversation_phase1_data, interview_phase2_data, conversation_phase2_data, all_data, vocab):
    numpy.save("preprocess/SEAME/arr/interview_phase1_data", interview_phase1_data)
    numpy.save("preprocess/SEAME/arr/interview_phase2_data", interview_phase2_data)
    numpy.save("preprocess/SEAME/arr/conversation_phase1_data", conversation_phase1_data)
    numpy.save("preprocess/SEAME/arr/conversation_phase2_data", conversation_phase2_data)
    numpy.save("preprocess/SEAME/arr/all_data", all_data)
    numpy.save("preprocess/SEAME/arr/vocab", vocab)