# ChatBot

## Introduction
In these challenging times, loneliness and self-isolation [has increased](https://www.sciencedirect.com/science/article/abs/pii/S0165178120312257), particularly in older adults.
We choose a chatbot that offers loneliness solutions as a topic for GNG5125 with the hope that it could contribute to the alleviation of this significant challenge.

## Data
For our project, we gathered data using web scraping from [A lonely life forum](https://www.alonelylife.com/forumdisplay.php?fid=4), where people chat, share their feelings, offer advice and solutions.

**Important note:** In no way we wanted to offend the original posters of the forum. With the utmost respect and from the bottom of our hearts, we wish you to find ways to deal with the challenges you face. We only hope that the data collected could be used to makes people's lives better.

## Thechnology
We use Gitflow and API that runs classification code written in Python. 

## Pre-requesties to build Chatbot API docker image
	1. Docker has to be installed
	https://hub.docker.com/editions/community/docker-ce-desktop-windows/
	2. Git
	https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
	3. Heroku CLI
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

## Setup DialogFlow fulfilment
https://www.youtube.com/watch?v=n4IPOeFCDxI