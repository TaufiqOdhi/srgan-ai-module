FROM taufiqodhi/srgan-ai-module:req
RUN mkdir app
WORKDIR /app
COPY . .

CMD ["python", "worker_dequeue.py"]