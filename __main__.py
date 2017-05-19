from csv import reader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import nltk

def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset


def get_artists(amount=None):
    artists = []
    for item in dataset:
        artists.append(item[0])
    if amount:
        return list(set(artists))[:amount]
    return list(set(artists))


def get_lyrics(item):
    return item[-1]


def get_songs_by_artist(artist):
    songs = []
    for item in dataset:
        if item[0] == artist:
            songs.append(item)
    return songs


def get_analysed_artist_songs(artist, amount=None):
    analysed_songs = {}
    songs = get_songs_by_artist(artist)
    for counter, song in enumerate(songs):
        if amount:
            if counter == amount:
                break
        analysed_songs[song[1]] = analyse_sentiment(song[-1])
    return analysed_songs


def get_artist_sentiment(artist):
    total = .0
    analysed_songs = get_analysed_artist_songs(artist)
    for song in analysed_songs:
        total += analysed_songs[song]  # ['compound']
    return total / len(analysed_songs)


def get_artists_sentiment(artists):
    artists_sentiment = {}
    for artist in artists:
        artists_sentiment[artist] = get_artist_sentiment(artist)
    return artists_sentiment


def analyse_sentiment(text, analyser='textblob'):
    if analyser == 'vader':
        return analyse_sentiment_vader(text)['compound']
    elif analyser == 'textblob':
        return analyse_sentiment_textblob(text)
    else:
        return (analyse_sentiment_vader(text)['compound'] + analyse_sentiment_textblob(text)) / 2
        

def analyse_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def analyse_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment = 0
    for sentence in blob.sentences:
        sentiment += sentence.sentiment.subjectivity
    return sentiment / len(blob.sentences)

def count_artist_verb_occurence(artist, amount=None):
    analysed_songs = {}
    songs = get_songs_by_artist(artist)
    for counter, song in enumerate(songs):
        if amount:
            if counter == amount:
                break
        analysed_songs[song[1]] = count_verbs(song[-1])
    return analysed_songs


def get_verb_analysed_artist_songs(artist, amount=None):
    analysed_songs = {}
    songs = get_songs_by_artist(artist)
    for counter, song in enumerate(songs):
        if amount:
            if counter == amount:
                break
        analysed_songs[song[1]] = count_verbs(song[-1])
    return analysed_songs


def get_artist_verb_occurence(artist):
    total = .0
    analysed_songs = get_verb_analysed_artist_songs(artist)
    for song in analysed_songs:
        total += analysed_songs[song]
    return total / len(analysed_songs)

def count_verb_occurence_artists(artists):
    artists_verb_occurence = {}
    for artist in artists:
        artists_verb_occurence[artist] = get_artist_verb_occurence(artist)
    return artists_verb_occurence


def count_verbs(text):
    total_verbs = 0
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tagged_text = nltk.pos_tag(text, tagset='universal')
    counts = Counter(tag for word,tag in tagged_text)
    
    return counts['VERB'] / len(tagged_text) * 100

def plot(dictionary, ylim=None):    
    plt.bar(range(len(dictionary)), list(dictionary.values()), color='yellow', align='center')
    plt.xticks(range(len(dictionary)), dictionary.keys(), rotation=90)
    if ylim:
        plt.ylim(ylim)
    plt.show()


dataset = load_csv('/home/dennis/Workspace/vu/TextMining/final_assignment/songdata.csv')

# example commands

# get a list of all the artists in the corpus
# print(get_artists())

# get the compound sentiment score of all the artist songs
# print(get_analysed_artist_songs('ABBA'), [-1,1])

# plot a graph of the mean sentiment of x=20 artists
# plot(get_artists_sentiment(get_artists(20)), [0,1])

# plot a graph of the mean percentage of verbs of x=20 artists
# plot(count_verb_occurence_artists(get_artists(20)))

# plot a graph of the sentiment of x=20 songs from an artist
# plot(get_analysed_artist_songs('Pitbull', 20), [-1, 1])

# plot a graph of the percentage of verbs from x=20 songs from an artist
# plot(count_artist_verb_occurence('ABBA', 20))

# print the song attributes for all the songs of an artist
# for song in get_songs_by_artist('ABBA'):
#     for attr in song:
#         print(attr)
