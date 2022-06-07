
"""
Created @author: AG
"""

import nltk


#pos_tweets = [('I love this car', 'positive'),
#                  ('This view is amazing', 'positive'),
#                  ('I feel great this morning', 'positive'),
#                  ('I am so excited about the concert', 'positive'),
#                  ('He is my best friend', 'positive')]

#neg_tweets = [('I do not like this car', 'negative'),
#                  ('This view is horrible', 'negative'),
#                  ('I feel tired this morning', 'negative'),
#                  ('I am not looking forward to the concert', 'negative'),
#                  ('He is my enemy', 'negative')]

pos_tweets = []
neg_tweets = []

with open('dataTrg.txt', 'r') as f:
  for line in f:
     line = line[:-1]
     parts = line.rsplit('\t', 2)
     sentence = parts[0].replace('.',' ')
     sentence = sentence.strip()
     sentiment = parts[1].strip()
     if sentiment == '1':
       pos_tweets.append((sentence, 'positive'))
     elif sentiment == '0':
       neg_tweets.append((sentence, 'negative'))

print(pos_tweets)
print(neg_tweets)


tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

print(tweets)


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features, wordlist


all_tweeted_words = get_words_in_tweets(tweets)
print(all_tweeted_words)

word_features, word_freq_dist = get_word_features(all_tweeted_words)

print(type(word_features))
print(word_features)
print(type(word_freq_dist))
print(word_freq_dist)

total = 0
number = 0
for word in word_freq_dist.keys():
    total = total + word_freq_dist.freq(word)
    number = number+1


print(number)
print(total)

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


training_set = nltk.classify.apply_features(extract_features, tweets)

classifier = nltk.NaiveBayesClassifier.train(training_set)
print(classifier.show_most_informative_features(256))

test_tweets = []
import random
with open("dataTest.txt","r") as file:
    for i in file.readlines():
        test_tweets.insert(int(random.random()*len(test_tweets)),i)

for tweet in test_tweets[:10]:
  print (tweet, "\t",     classifier.classify(extract_features(tweet.split())))        
         


