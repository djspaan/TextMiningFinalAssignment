# TextMiningFinalAssignment
The Final Assignment for the Text Mining course of the VU Amsterdam.

# Example commands

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
