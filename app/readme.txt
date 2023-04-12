App is hosted on 0.0.0.0:8080

To access the app, create an environment, install requirements, then run app.
- python -m venv .venv
- pip install -r requirements.txt
- python app.py

To build app through docker, create container with Dockerfile
- docker build -f Dockerfile -t nlp-dashboard .
- docker run -p 8080:8080 nlp-dashboard

Thanks for checking out our app!

- Daniel Setiawan -- www.linkedin.com/in/danielosetiawan
- Laurel He -- www.linkedin.com/in/cheng-laurel-he-b04a59104

### Internal
To deploy to GCP:
- GCP console >>>
- git clone https://github.com/danielosetiawan/NLP_sentiment_analysis.git
- docker build -f Dockerfile -t gcr.io/<project_id>/nlp-dashboard .
- docker push gcr.io/nlp-dashboard-383105/nlp-dashboard
- gcloud run deploy nlp-db --image=gcr.io/nlp-dashboard-383105/nlp-dashboard --platform=managed --region=us-west2 --cpu=4 --memory=16G --allow-unauthenticated