from flask import Flask
import Processor as processor
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"


@app.route('/<text>')
def hello_name(text):
    book_label = processor.getBook(text)
    return "Text from the book {}".format(book_label)

if __name__ == '__main__':
    processor.Init()
    app.run()

