import gensim.downloader as api
import pandas as pd
import sys
import random
import csv
import numpy as np

def write_to_csv(file_name,header, rows):
    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def AI_Answer(model, input_file_path, model_name):
    csv_file = pd.read_csv(input_file_path)
    question_words = csv_file.question
    answer = csv_file.answer
    csv_file.drop('question', inplace=True, axis=1)
    csv_file.drop('answer', inplace=True, axis=1)
    guesses = csv_file.values
    rows_to_write = list(list())

    for i in range(len(question_words)):
        guess_rank = -sys.maxsize
        for j in range(len(guesses[0])):
            try:
                sim = model.similarity(question_words[i], guesses[i][j])
                if guess_rank < sim:
                    guess_rank = sim
                    guessed_word = guesses[i][j]
                if guessed_word == answer[i]:
                    label = 'correct'
                else:
                    label = 'wrong'
            except:
                guessed_word = guesses[i][random.choice(range(len(guesses[0])))]
                label = 'guess'
        rows_to_write.append([question_words[i], answer[i], guessed_word, label])
        headers = ['Question', 'Solution', 'AI Guess', 'Label']
    write_to_csv(f'{model_name}-details.csv', headers, rows_to_write)

def Analysis(model, model_result_path):
    model_name = model_result_path.split("-details")[0]
    vocabulary_size = len(model.index_to_key)
    csv_file = pd.read_csv(model_result_path)
    labels = np.asarray( csv_file.Label)
    C = np.count_nonzero(labels == "correct")
    V = np.count_nonzero(labels == "wrong") + C
    accuracy = C/V
    row = [model_name, vocabulary_size, C, V, accuracy]
    headers = ['Model Name', 'Vocabulary Size', 'Correct', 'Non Guessed', 'Accuracy']
    write_to_csv('analysis.csv', headers, [row])


# TASK 1
wv = api.load('word2vec-google-news-300')
input_file_path = './synonyms.csv'
AI_Answer(wv, input_file_path, 'word2vec-google-news-300')
Analysis(wv, 'word2vec-google-news-300-details.csv')

# TASK 2
# 2 Different Corpus with the same embedded size of 100
wv_twitter_100 = api.load("glove-twitter-100")
input_file_path = './synonyms.csv'
AI_Answer(wv_twitter_100, input_file_path, "glove-twitter-100")
Analysis(wv_twitter_100, "glove-twitter-100-details.csv")	

wv_wiki_100 = api.load("glove-wiki-gigaword-100")
input_file_path = './synonyms.csv'
AI_Answer(wv_wiki_100, input_file_path, "glove-wiki-gigaword-100")
Analysis(wv_wiki_100, "glove-wiki-gigaword-100-details.csv")

# 2 Same Corpus with different embedded size
wv_twitter_25 = api.load("glove-twitter-25")
input_file_path = './synonyms.csv'
AI_Answer(wv_twitter_25, input_file_path, "glove-twitter-25")
Analysis(wv_twitter_25, "glove-twitter-25-details.csv")	

wv_twitter_50 = api.load("glove-twitter-50")
input_file_path = './synonyms.csv'
AI_Answer(wv_twitter_50, input_file_path, "glove-twitter-50")
Analysis(wv_twitter_50, "glove-twitter-50-details.csv")	



