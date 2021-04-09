# ChatBot

## Pre-requesties to build API docker image
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

## Using curl to query API
You can run the code locally using 
```
python NLPWebserver.py
```
then query the API 
```
curl http://127.0.0.1:5000/
```

## Setup DialogFlow fulfilment
https://www.youtube.com/watch?v=n4IPOeFCDxI

## Hiroku ChatBot API deployment
```
heroku login
heroku container:push web --app=gng5125t10-chatbot-api
heroku container:release web --app=gng5125t10-chatbot-api
```
To see logs:
```
heroku logs --tail --app=gng5125t10-chatbot-api
```
Also see:
https://devcenter.heroku.com/articles/container-registry-and-runtime