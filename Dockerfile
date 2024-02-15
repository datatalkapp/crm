FROM python:3.12
ENV PYTHONUNBUFFERED=1
ENV PYTHONPYCACHEPREFIX=/tmp
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
VOLUME ./data /app/data
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "--bind", "0.0.0.0:8080", "app:app"]