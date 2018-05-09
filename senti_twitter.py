'''
;;; Multi_class Sentiment Analysis on Twitter using Machine learning model
;;; Copyright © 2018 Uday Kiran Kondreddy <udaykiran.kondreddy@gmail.com>
;;; Copyright © 2018 Farhaan Ahmed Shaik <farhaanfsk@gmail.com>
;;; Copyright © 2018 Subhani Kurra <subhanikurra4@gmail.com>
;;; Copyright © 2018 Naga Teja Mamidapaka <nagatejam@gmail.com>
;;;
;;; This file is part of Sentimental Analysis on Twitter.
;;;
;;; This is free Code; you can redistribute it and/or modify it
;;; under the terms of the GNU General Public License as published by
;;; the Free Software Foundation; either version 3 of the License, or (at
;;; your option) any later version.
;;;
;;; This code is distributed in the hope that it will be useful, but
;;; WITHOUT ANY WARRANTY; without even the implied warranty of
;;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;;; GNU General Public License for more details.
;;;
;;; See <http://www.gnu.org/licenses/>.'''

import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from nltk import word_tokenize
from nltk.corpus import stopwords
 
class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = 'XXXXXXXXXXXXXXXXXX'
        consumer_secret = 'XXXXXXXXXXXXXXX'
        access_token = 'XXXXXXX-XXXXXXXXXXXXXX'
        access_token_secret = 'XXXXXXXXXXXXXXXXXX'
 
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
 
    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)",'', tweet).split())
        
    def remove_urls(self, tweets):
        return re.sub("http.?://[^\s]+[\s]?",'',tweets) 
    
    def emotion_finder(self,sentence):
        stop = set(stopwords.words('english'))		
        happy= ["contented","content","cheerful","happy","won", 
				"cheery","merry","joyful","jovial",
				"jolly","joking","jocular","gleeful", 
				"carefree","untroubled","delighted", 
				"smiling","beaming","grinning","glowing", 
				"satisfied","gratified","buoyant","radiant","lol","happiness"]
				 
        sad= [	"unhappy","sorrowful","dejected","regretful","depressed", 
				"downcast","miserable","downhearted","down","despondent","despairing", 
				"disconsolate","gloomy","funeral"]			
				
        angry= ["angry","rage","annoyed","cross","vexed","irritated","exasperated","irritate","temper" 
				"indignant","aggrieved", "irked", "piqued", "displeased", "provoked", 
				"galled", "resentful", "furious", "enraged", "infuriated", "in a temper",
				"incensed", "raging", "incandescent", "wrathful", "fuming", "ranting","annoying","annoy"]
				
        surprise= ["shock", "thunderbolt", "bombshell", "revelation",
				"source of amazement", "rude awakening", "eye-opener",  
				"turn up for the books", "shocker", "whammy"]	
        list = [i for i in sentence.lower().split() if i not in stop]					
        h1=s1=a1=su=0    				
        for i in list:
            for j in happy:	
                if(i == j):	    
                    h1+=1

            for j in sad:
                if(i==j):
                    s1+=1		
			
            for j in angry:
                if(i==j):
                    a1+=1
			
            for j in surprise:
                if(i==j):
                    h1+=1
			

        if(h1>s1 and h1>a1 and h1>su): 
            print("the emotion of sentencee is happy")
        elif(s1>h1 and s1>a1 and s1>su):
            print("sad emotion")
        elif(a1>h1 and a1>s1 and a1>su):
            print("angry emotion")
        elif(su>h1 and su>s1 and su>a1):
            print("surprised")
        else:
            print("neutral")
                                   
 
    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        #print(analysis.sentiment)
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
 
    def get_tweets(self, query, count = 10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []
 
        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q = query, count = count)
 
            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}
 
                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
 
                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
 
            # return parsed tweets
            return tweets
 
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
 
def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    print("\n\t===> Enter the keyword you want to analyze on twitter:\n")
    topic= input()
    # calling function to get tweets
    tweets = api.get_tweets(topic , count = 500)
    print("\t\t\t\t\tSTARTS HERE")
    
    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # percentage of positive tweets
    print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # percentage of negative tweets
    print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
    # percentage of neutral tweets
    #print("Neutral tweets percentage: {} %".format(100*len(tweets - ntweets - ptweets)/len(tweets)))
        
 
    # printing first 5 positive tweets
    print("\n\nPositive tweets:")
    for tweet in ptweets[:10]:
        psentence=api.remove_urls(tweet['text'])
        print(psentence)
        print("\n")        
        api.emotion_finder(psentence)
    
    # printing first 5 negative tweets
    print("\n\nNegative tweets:")
    for tweet in ntweets[:10]:
        nsentence=api.remove_urls(tweet['text'])
        print(nsentence)
        print("\n")
        api.emotion_finder(nsentence)                
 
if __name__ == '__main__':
    main()
