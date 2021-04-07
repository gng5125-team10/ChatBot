from flask import Flask, request, jsonify, json
import Processor as processor
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
#import ssl
#import os.path

app = Flask(__name__)
auth = HTTPBasicAuth()

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
        print("Received text:", text)
        book_label = svm.PredictLabel(text)
        return_text = "Text from the book {}".format(book_label)

        return jsonify(
        status=200,
        replies=[
            {
            'fulfillmentMessages': [
                {
                'text': {
                    'text': [
                        return_text
                    ]
                }
                }
            ]
            }
        ]
        )

    except:
        print("error decoding", request.get_data())
        return jsonify(
        status=400,
        replies=[]
        )


if __name__ == '__main__':
    processor.Init()

    #for http
    app.run()

    
    #for https
    #context = ssl.SSLContext()
    #cert1 = "cert.pem"
    #cert2 = "key.pem"
    #print (os.getcwd())
    #context.load_cert_chain(cert1 , cert2)
    #app.run(ssl_context=context)