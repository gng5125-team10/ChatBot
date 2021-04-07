# ChatBot

## Pre-requesties to build API docker image
	1. docker has to be installed
	https://hub.docker.com/editions/community/docker-ce-desktop-windows/
	2. git
	https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

## To Copy the code locally
```
git clone https://github.com/gng5125-team10/ChatBot.git
```
Add /modify files (the text in brackets is commit message)
```
git commit -am "Adding first version of docker image"
git push
```

## Flusk Python webserver
We are using Flusk wev-server to implement the Chatbot backend API:
https://github.com/pallets/flask/tree/master/tests
https://realpython.com/flask-by-example-part-1-project-setup/

## Dockerfile for Python Flusk
https://docs.docker.com/compose/gettingstarted/

## Build and run Docker container
```
docker build --no-cache -t chatbotapi ./code/nlpapi
docker run  -p 5000:5000  chatbotapi
```

## Using curl to query API
```
curl https://api.datamuse.com/words?sp=t??k
curl http://127.0.0.1:5000/test%20wfjsdofn%20a%20wolnot
```

## Deploy docker image to Google Cloud
https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app




## Model Persistence
https://scikit-learn.org/stable/modules/model_persistence.html


## Using Python Freeze to compile python files
https://wiki.python.org/moin/Freeze


## Deploy to the Cloud - Google Kubernetes Engine (GKE)
1. Install Google Cloud SDK
https://cloud.google.com/sdk/docs/install
1.a Add docker authentification
```gcloud auth configure-docker```

2. Enable Container registry
https://cloud.google.com/container-registry/docs/pushing-and-pulling

3. Tag and push the image (in cloud shell)
```
docker tag chatbotapi gcr.io/gng5125t10chatbotapi/chatbotapi:v0.4
docker push gcr.io/gng5125t10chatbotapi/chatbotapi:v0.0
```

4.  Deploying container in GKE cluster
````
 gcloud config set project gng5125t10chatbotapi
 gcloud config set compute/zone northamerica-northeast1-b
````

5. Enable API service - container.googleapis.com, execute command:

````
	gcloud services enable container.googleapis.com 
`````

6. Creating a GKE cluster
````
 gcloud container clusters create chatbotapi --num-nodes=1
 gcloud container clusters get-credentials chatbotapi
````
 The second command configures kubectl to use the cluster you created.

7. Deploying an application to the cluster
```` 
 kubectl create deployment chatbot-api --image=gcr.io/gng5125t10chatbotapi/chatbotapi:v0.0
````
8. Exposing the deployment to the world
````
kubectl expose deployment chatbot-api --type LoadBalancer --port 5000 --target-port 5000
````

After service(deployment) is created 
````
kubectl get service
NAME          TYPE           CLUSTER-IP     EXTERNAL-IP    PORT(S)          AGE
chatbot-api   LoadBalancer   10.35.242.19   34.95.48.224   5000:32673/TCP   48s
kubernetes    ClusterIP      10.35.240.1    <none>         443/TCP          14m
````
Where ```34.95.48.224``` is IP of the service

9. To delete
````
 delete deployment chatbot-api
 delete service chatbot-api
````


## Setup DialogFlow fulfilment
https://www.youtube.com/watch?v=n4IPOeFCDxI

## Setting-up HTTPs via ngnx server
https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https
https://medium.com/analytics-vidhya/how-to-deploy-a-python-api-with-fastapi-with-nginx-and-docker-1328cbf41bc