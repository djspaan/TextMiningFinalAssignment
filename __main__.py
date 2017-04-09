from csv import reader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


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
        analysed_songs[song[1]] = analyse_sentiment(song[-1])['compound']
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


def analyse_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def plot(dictionary):
    plt.bar(range(len(dictionary)), dictionary.values(), align='center')
    plt.xticks(range(len(dictionary)), dictionary.keys(), rotation=90)
    plt.ylim([-1, 1])
    plt.show()


dataset = load_csv('/home/dennis/Workspace/vu/TextMining/final_assignment/songdata.csv')

# example commands

# get a list of all the artists in the corpus
print(get_artists())

# get the compound sentiment score of all the artist songs
print(get_analysed_artist_songs('ABBA'))

# plot a graph of the mean sentiment of x=20 artists
plot(get_artists_sentiment(get_artists(20)))

# plot a graph of the sentiment of x=20 songs from an artists
plot(get_analysed_artist_songs('Bob Marley', 20))

# print the song attributes for all the songs of an artist
for song in get_songs_by_artist('ABBA'):
    for attr in song:
        print(attr)
