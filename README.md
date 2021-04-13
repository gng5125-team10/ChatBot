# ChatBot

## Introduction
In these challenging times, loneliness and self-isolation [has increased](https://www.sciencedirect.com/science/article/abs/pii/S0165178120312257), particularly in older adults.
We choose a chatbot that offers loneliness help as a topic for GNG5125 with the hope that it could contribute to the alleviation of this significant challenge.

## Data
For our project, we gathered data using web scraping from [A lonely life forum](https://www.alonelylife.com/forumdisplay.php?fid=4), where people chat, share their feelings, offer advice and solutions.
We extracted \~40 000 posts (\~10 000 posts and \~30 000 responses)\
**Important note:** In no way we wanted to offend the original posters of the forum. With the utmost respect and from the bottom of our hearts, we wish you to find ways to deal with the challenges you face. We only hope that the data collected could be used to makes people's lives better.

## Technology
We use Gitflow and API that runs classification code written in Python. 

## Feature Engineering
We used and compared several NLP feature extraction techniques:
* TD-IDF
* Word2Vec

## Unsupervised learning
During the unsupervised learning step we clustered the posts and determined the number of clusters the fits the data the best.
To run clustering:
```
python ./code/clustering.py
```

## Supervised learning
During the supervised we train and save model for use by the classification.
To train model:
```
python ./code/model_training.py
```

## Pre-requesties to build Chatbot API docker image
	1. Docker has to be installed
	https://hub.docker.com/editions/community/docker-ce-desktop-windows/
	2. Git
	https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
	3. Heroku CLI (to deploy to Heroku cloud)
	https://devcenter.heroku.com/articles/heroku-cli

## To Copy the code locally
```
git clone https://github.com/gng5125-team10/ChatBot.git
```
Add /modify files (the text in brackets is commit message)
```
git commit -am "Adding first version of docker image"
git push
```

## To Run Code locally
From the Chatbot folder execute:
```
python ./code/NLPWebserver.py
```
while the server is running you can query the API using curl, for example:
```
curl http://127.0.0.1:5000/
```

## Hiroku ChatBot API deployment
In our "production" we use the API deployed to Heroku cloud.
The following commands deploy Docker image (from ./code folder):
```
heroku login
heroku container:login
heroku container:push web --app=gng5125t10-chatbot-api
heroku container:release web --app=gng5125t10-chatbot-api
```
To see logs:
```
heroku logs --tail --app=gng5125t10-chatbot-api
```
Also see:
https://devcenter.heroku.com/articles/container-registry-and-runtime

## Chatbot interface
https://gng5125-team10.github.io/


