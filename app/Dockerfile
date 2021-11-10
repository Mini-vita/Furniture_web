FROM python:3.8
WORKDIR /home/daewonng12/apache-dockerfile 
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
COPY requirements.txt /home/daewonng12/apache-dockerfile/requirements.txt
COPY style_all.csv /home/daewonng12/apache-dockerfile/style_all.csv
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 8080
COPY . /home/daewonng12/apache-dockerfile
CMD streamlit run --server.port 8080 --server.enableCORS false app.py

