FROM python:3.9

WORKDIR PCB_Demo

RUN pip install --upgrade pip==20.2
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY checkpoint ./checkpoint/
COPY dataset ./dataset/
COPY images ./images/
COPY model ./model/
COPY main.py ./
COPY utils.py ./
#--server.port 8503
CMD ["streamlit", "run", "./PCB_Demo/main.py", "--server.port 8503"]
EXPOSE 8503

#to build image from dockerfile
#docker build -t pcb-analysis-demo .

#run
#docker run -p 8503:8503 pcb-analysis-demo streamlit run main.py --server.port 8503