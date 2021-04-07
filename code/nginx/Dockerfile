FROM nginx

#COPY default /etc/nginx/sites-enabled/default
COPY default  /etc/nginx/conf.d/default.conf
#COPY nginx.conf /etc/nginx/nginx.conf
#COPY nginx.conf /etc/nginx/conf.d/default.conf
#COPY cert.pem /etc/letsencrypt/live/api.domain.com/fullchain.pem
#COPY key.pem /etc/letsencrypt/live/api.domain.com/privkey.pem
#COPY options-ssl-nginx.conf /etc/letsencrypt/options-ssl-nginx.conf
#COPY ssl-dhparams.pem /etc/letsencrypt/ssl-dhparams.pem

RUN apt-get install -y curl