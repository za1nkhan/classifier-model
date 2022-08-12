FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./mnbmodel.pkl /code/

COPY ./vector.pkl /code/

RUN pip install -r /code/requirements.txt

RUN python -m pip install requests

RUN pip install numpy

RUN pip install fastapi uvicorn

RUN pip install sklearn

#RUN pip install joblib

COPY ./deploy.py /code/deploy.py

EXPOSE 8002

#ENTRYPOINT ["uvicorn", "deploy:app --reload"]

#CMD ["uvicorn", "code.deploy:app", "8002"]

#CMD python -m uvicorn code.deploy:deploy --host 0.0.0.0 --port 8002

CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8002"]

