FROM python:3.7
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.doubanio.com/simple/
COPY . .
ENV PYTHONPATH=/usr/src/app:$PYTHONPATH
CMD ["python","./core/predict.py"]
