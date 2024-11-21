import pandas as pd
import numpy as np
import re, string
import os 
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import linear_kernel

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Load Google Universal sentence Encoder(DAN) Pretrained model

#! curl -L -o 4.tar.gz "https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed" 
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# module_path ="/home/zettadevs/GoogleUSEModel/USE_4"
model = hub.load(module_url)
#print ("module %s loaded" % module_url)

#Create function for using modeltraining
def embed(input):
    return model(input)

# Use Case of Google USE
def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Similarity")

def run_and_plot_U(messages_):
    message_embeddings_ = embed(messages_)
    plot_similarity(messages_, message_embeddings_, 90)

def SimilarityScore(messages):
    message_embedding = embed(messages)
    corr = np.inner(message_embedding,message_embedding)
    # for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    #    print("Message: {}".format(messages[i]))
    print(corr)

# Use Case 1: Word Semantic
WordMessage = ['big data', 'millions of data', 'millions of records','cloud computing','aws','azure','saas','bank','account']
run_and_plot_U(WordMessage) 

# Use Case 2: Sentence Semantic
SentMessage = ['How old are you?','what is your age?','how are you?','how you doing?']
run_and_plot_U(SentMessage)

# Use Case 3: Word, Sentence and Paragram Semantic
word ='Cloud computing'
Sentence = 'what is cloud computing'
Para = (
    "Cloud computing is the latest generation technology with a high IT infrastructure that provides us a means by which we can use and utilize the applications as utilities via the internet."
    "Cloud computing makes IT infrastructure along with their services available 'on-need' basis." 
    "The cloud technology includes - a development platform, hard disk, computing power, software application, and database.")
Para5 = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be."
    )
Para6 = (
    "Azure is a cloud computing platform which was launched by Microsoft in February 2010."
    "It is an open and flexible cloud platform which helps in development, data storage, service hosting, and service management."
    "The Azure tool hosts web applications over the internet with the help of Microsoft data centers.")
case4Message=[word,Sentence,Para,Para5,Para6]

SimilarityScore(case4Message)