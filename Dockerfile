FROM python:3.6-slim 

COPY requirements.txt /

RUN pip install -r /requirements.txt && cd Deploy

COPY . .

ENTRYPOINT [ "python3" ]

CMD ["app.py"]
