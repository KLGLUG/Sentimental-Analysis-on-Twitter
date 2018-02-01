import twitter
import json
import csv
import re
import time
from urllib.parse import unquote

query = input("Hashtag:")
consumer_key = 'A0PIowPC2WDf965TSlw5THleS'
consumer_secret = 'bECpVLYsu3aINGRSqPFkyD2xE387I5Yw6XayabXVntFEvNV4Lr'
access_token = '2348573101-ltHO8PVdOH9TOecf6F448w9D0jSn3riaf4Xdpgu'
access_token_secret = 'y1eGbgZwgxT8hmvqjiVymMDBVHjeWp0SK4ghYE671he3D'



auth = twitter.oauth.OAuth(access_token, access_token_secret,consumer_key,consumer_secret)
twitter_api = twitter.Twitter(auth=auth)
max_count = int(input("Count:"))
search_results = {}
search_results["statuses"] = []
search_results["search_metadata"] = {}
search_results = twitter_api.search.tweets(q=query,count=max_count,lang = 'en')
statuses = search_results['statuses']
while(1):
    if len(statuses)>=max_count:
        break
    if 'next_results' in search_results['search_metadata'].keys() and 'search_metadata' in search_results.keys():
        next_results = search_results['search_metadata']['next_results']
    else:
        break
    kwargs = dict([ kv.split('=') for kv in unquote(next_results[1:]).split("&") ])
    try:
        search_results = twitter_api.search.tweets(**kwargs)
    except Exception as e:
        time.sleep(910)
        print('Slept, Continuing after exception:')
        search_results = twitter_api.search.tweets(**kwargs)
    statuses += search_results['statuses']
with open(query+str(int(time.time()))+'.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Tweet", "Emotion"])
    for status in statuses:
        if 'RT' not in status["text"]:
            status["text"] = re.sub(r"(?:\@|https?\://)\S+", "", status["text"])
            writer.writerow([status["text"],""])
