FROM python:3.9-slim-buster


WORKDIR /code
ENV FLASK_APP=NLPWebserver.py
ENV FLASK_RUN_HOST=0.0.0.0


ENV PYTHON_PACKAGES="\
	flask \
	contractions \
	nltk \ 
	numpy \
	pandas \
	pillow \
	scikit-learn \
	wordcloud \
	pyopenssl \
    " 
RUN apt-get update && apt-get -y install gcc
RUN pip install --no-cache-dir $PYTHON_PACKAGES


RUN python -m nltk.downloader punkt && \
     python -m nltk.downloader stopwords && \
 	 python -m nltk.downloader wordnet && \
 	 python -m nltk.downloader gutenberg


EXPOSE 5000
COPY . .
CMD ["flask", "run"]