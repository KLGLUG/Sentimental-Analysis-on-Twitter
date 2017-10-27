import tweepy
from textblob import TextBlob

consumer_key = '0TI38y7MiWGozjh27xq3juY8s'
consumer_secret = 'DERgSRYujeUeuUJ7unuWgkXRevMftm15Vo4N4cigxZnuhPkJD7'

access_token = '624916821-nLo973hLFNf5JemrKTOkkZY9aOuE2OqcO5j5IswV'
access_token_secret = 'IwhBILv2Kcenw88ea3QOqUkJfYnFzow5PMrAopYO7cR1C'

#access to my no good twitter app

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
		
print("\n\t===> Enter the keyword you want to analyze on twitter:\n")
topic= input()
public_tweets = api.search(topic)
print("\t\t\t\t\tSTARTS HERE")
ptweets=ntweets=0

for tweet in public_tweets:
    print("")
    print("Analysis Below:\n\n")
    #encoded the tweets in utf-8 to dodge errors
    print(tweet.text.encode('utf-8'))
    
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    #here we can determine whether the tweeter has good opinion or bad opinion or neutral opinion
    if analysis.sentiment.polarity > 0.0:
    	print('Positive sentiment')
    elif analysis.sentiment.polarity == 0.0:
        print('Nuetral Sentiment')
    else:
        print("Negative Sentiments")
    #here we can determine if the tweet is objective or subjective
    if analysis.sentiment.subjectivity <= 0.5:
        print('Most Likely Factual')
    elif analysis.sentiment.subjectivity > 0.5:
        print('Least Likely Factual')
    print("")
    print(ptweets)
    print(ntweets)
print("\t\t\t\t\tENDS HERE")
