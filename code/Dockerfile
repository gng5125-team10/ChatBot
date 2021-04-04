#FROM python:3.6-alpine
FROM python:3.6

WORKDIR /code
ENV FLASK_APP=NLPWebserver.py
ENV FLASK_RUN_HOST=0.0.0.0

#For alpine
#RUN apk add build-base



ENV PYTHON_PACKAGES="\
	flask \
	contractions \
	nltk \ 
	numpy \
	pandas \
	pillow \
	scikit-learn \
	wordcloud \
    " 

RUN pip install --no-cache-dir $PYTHON_PACKAGES


RUN python -m nltk.downloader punkt && \
     python -m nltk.downloader stopwords && \
 	 python -m nltk.downloader wordnet && \
 	 python -m nltk.downloader gutenberg


EXPOSE 5000
COPY . .
CMD ["flask", "run"]