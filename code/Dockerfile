FROM python:3.9-slim-buster

ENV PYTHON_PACKAGES="\
	flask \
	contractions \
	nltk \ 
	numpy \
	pandas \
	pillow \
	pytest \
	scikit-learn \
	wordcloud \
	flask-httpauth\
	gunicorn \
	Jinja2 \
	Werkzeug \
    " 
RUN apt-get update && apt-get -y install gcc
RUN pip install --no-cache-dir $PYTHON_PACKAGES

COPY . .

##copy nltk corpora to the container
RUN apt-get -y install unzip
COPY ./nltk_data/corpora.zip /usr/local/nltk_data/
RUN cd /usr/local/nltk_data/ && unzip corpora.zip && rm corpora.zip

#Run locally
#EXPOSE 5000
#CMD ["flask", "run"]

# Run on Heroku			
CMD gunicorn --bind 0.0.0.0:$PORT wsgi 