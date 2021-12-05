import gensim.downloader as api
import pandas as pd
import sys
import random
import csv
import numpy as np

def make_header(file_name,header):
    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(header)

def write_to_csv(file_name,rows):
    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def ai_answer(model, input_file_path, model_name):
    count = 0;
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
    write_to_csv(f'{model_name}-details.csv', rows_to_write)

def analysis(model, model_result_path):
    model_name = model_result_path.split("-details")[0]
    vocabulary_size = len(model.index_to_key)
    csv_file = pd.read_csv(model_result_path)
    labels = np.asarray( csv_file.Label)
    C = np.count_nonzero(labels == "correct")
    V = np.count_nonzero(labels == "wrong") + C
    accuracy = C/V
    row = [model_name, vocabulary_size, C, V, accuracy]
    write_to_csv('analysis.csv', [row])



header_ai_answers = ['Question', 'Solution', 'AI Guess', 'Label']
header_analysis = ['Model Name', 'Vocabulary Size', 'Correct', 'Non Guessed', 'Accuracy']
input_file_path = './synonyms.csv'
make_header('analysis.csv',header_analysis)
# TASK 1
wv = api.load('word2vec-google-news-300')
make_header('word2vec-google-news-300-details.csv',header_ai_answers)
ai_answer(wv, input_file_path, 'word2vec-google-news-300')
analysis(wv, 'word2vec-google-news-300-details.csv')

# TASK 2
# 2 Different Corpus with the same embedded size of 100
wv_twitter_100 = api.load("glove-twitter-100")
make_header("glove-twitter-100-details.csv",header_ai_answers)
ai_answer(wv_twitter_100, input_file_path, "glove-twitter-100")
analysis(wv_twitter_100, "glove-twitter-100-details.csv")	

wv_wiki_100 = api.load("glove-wiki-gigaword-100")
make_header("glove-wiki-gigaword-100-details.csv",header_ai_answers)
ai_answer(wv_wiki_100, input_file_path, "glove-wiki-gigaword-100")
analysis(wv_wiki_100, "glove-wiki-gigaword-100-details.csv")

# 2 Same Corpus with different embedded size
wv_twitter_25 = api.load("glove-twitter-25")
make_header("glove-twitter-25-details.csv",header_ai_answers)
ai_answer(wv_twitter_25, input_file_path, "glove-twitter-25")
analysis(wv_twitter_25, "glove-twitter-25-details.csv")	

wv_twitter_50 = api.load("glove-twitter-50")
make_header("glove-twitter-50-details.csv",header_ai_answers)
ai_answer(wv_twitter_50, input_file_path, "glove-twitter-50")
analysis(wv_twitter_50, "glove-twitter-50-details.csv")	



