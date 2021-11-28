import json
import random
import tensorflow
import tflearn
import numpy as np
import nltk
import pickle
import os.path
from nltk.stem.lancaster import LancasterStemmer
from tflearn.layers.core import activation
stemmer = LancasterStemmer()

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load()
    with open("intents.json") as f:
        data = json.load(f)

except:
    with open("intents.json") as f:
        data = json.load(f)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            pattern_words = nltk.word_tokenize(pattern)
            pattern_words = [stemmer.stem(w.lower()) for w in pattern_words]
            words.extend(pattern_words)
            docs_x.append(pattern_words)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = sorted(list(set(words)))  # set to remove doubles
    words.remove("?")  # because we don't want any question mark in our model
    words.remove("!")  # same for exclamation mark
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for i, doc in enumerate(docs_x):
        bag = []

        for w in words:
            if w in doc:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[i])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
        f.close()

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#  check if a model already exists
if not(os.path.exists("./model.tflearn.meta")) or not(os.path.exists("./model.tflearn.index")):
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

else:
    model.load("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]

    for s_word in s_words:
        for i, w in enumerate(words):
            if s_word == w:
                bag[i] = 1

    return np.array(bag)


def chat():
    print("\n------------------------------------------\n\nStart talking with the bot ! Ask any question about the WACS\nType quit to stop")
    while True:
        inp = input("You : ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)

        if results[0][results_index] > 0.7:
            tag = labels[results_index]

            for intent in data["intents"]:
                if intent["tag"] == tag:
                    responses = intent["responses"]
            print(random.choice(responses))

        else:
            print("I don't understand, please try again")


chat()
