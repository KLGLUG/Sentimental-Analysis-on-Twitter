# Sentimental-Analysis-on-Twitter

Sentiment Analysis is the process of ‘**computationally**’ determining whether a piece of writing is positive, negative or neutral. It’s also known as opinion mining, deriving the opinion or attitude of a speaker.

Sentiment analysis deals with identifying and classifying opinions or sentiments expressed in source text. Social media is generating a vast amount of sentiment rich data in the form of tweets, status updates, blog posts etc. Sentiment analysis of this user generated data is very useful in knowing the opinion of the crowd. Twitter sentiment analysis is difficult compared to general sentiment analysis due to the presence of slang words and misspellings. The maximum limit of characters that are allowed in Twitter is 140.

### Why sentiment analysis?

* **Business:** In marketing field companies use it to develop their strategies, to understand customers’ feelings towards products or brand, how people respond to their campaigns or product launches and why consumers don’t buy some
products.

* **Politics:** In political field, it is used to keep track of political view, to detect consistency and inconsistency between statements and actions at the government level. It can be used to predict election results as well!

* **Public Actions:** Sentiment analysis also is used to monitor and analyse social phenomena, for the spotting of potentially dangerous situations and determining the general mood of the blogosphere.




### Instructions :
```sh
# Install dependencies
$ pip3 install -r requirements.txt
# To run the Word2vec 
$ python3 senti_with_avg_word2vec.py
# To run TF-IDF word2vec
$ python3 senti_with_avg_tfidf_word2vec.py

```

Now in Python3 Terminal 
*   import nltk
*   nltk.download(“stopwords”)
*   nltk.download(“punkt”)
*   nltk.corpus()

IF you are intrested to download tweets on particular hashtag (#) 

https://github.com/KLGLUG/Tweet-Crawler
