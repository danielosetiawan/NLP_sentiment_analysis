FROM python:3.8
ENV DASH_DEBUG_MODE True

COPY ./data/ /data/
COPY ./content.py /content.py
COPY ./globals.py /globals.py
COPY ./sentiment_prediction.py /sentiment_prediction.py
COPY ./callbacks.py /callbacks.py
COPY ./data.py /data.py
COPY ./app.py /app.py
COPY ./img /img
COPY ./assets /assets
COPY ./logos /logos
COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN set -ex && \
    pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "app.py"]
