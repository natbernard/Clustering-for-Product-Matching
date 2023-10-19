FROM eu.gcr.io/iprocure-edw/data-cleanup:0.0.4 

WORKDIR /app

COPY . .

RUN pip install --upgrade pip setuptools wheel build

RUN pip install -r requirements.txt

RUN python -m build

RUN pip install ./dist/data-cleaning-0.0.1.tar.gz
