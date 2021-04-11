from flask import Flask, request, jsonify, json
import Processor as processor
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import logging

#import ssl
#import os.path

app = Flask(__name__)
auth = HTTPBasicAuth()

#logger = logging.getLogger('NLPWebserver')
#logger.setLevel(logging.DEBUG)

users = {
    "dialogflow": generate_password_hash("!WhatAGreatDay!"),
}

@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username


svm = processor.model_SVM_TFIDF()

@app.route('/')
def hello():
    return "Hello from Chatbot API"


@app.route('/bot', methods = ['POST', 'GET'])
@auth.login_required
def authorship():

    try:
        bot_data = json.loads(request.get_data())
        text = bot_data['queryResult']['queryText']
        #TODO Change loglevel to info (we use warning now so the messages appear by default)
        logging.warning("Received text: " + text)
        #TODO: may need to incorporate context here
        label = svm.PredictLabel(text)
        response = svm.getResponseForALabel(label)
        

        logging.warning("Response: " + response)

        return jsonify(
        status=200,
        fulfillmentMessages= [
                {
                'text': {
                    'text': [
                        response
                    ]
                }
                }
            ]

        )

    except:
        logging.warning("error decoding", request.get_data())
        return jsonify(
        status=400,
        replies=[]
        )


if __name__ == '__main__':
    processor.Init()

    #for http
    #app.run(host='0.0.0.0')
    app.run(host='0.0.0.0')
    
    #for https
    #context = ssl.SSLContext()
    #cert1 = "cert.pem"
    #cert2 = "key.pem"
    #print (os.getcwd())
    #context.load_cert_chain(cert1 , cert2)
    #app.run(ssl_context=context)