FROM python:3.6

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN cd Deploy

ENTRYPOINT [ "python3" ]

CMD ["app.py"]
