FROM python:3.12.0


WORKDIR /app

COPY ./requirements.txt ./requirements.txt

COPY . .

ENV FLASK_APP=__init__.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "flask","run" ]