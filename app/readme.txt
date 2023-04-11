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