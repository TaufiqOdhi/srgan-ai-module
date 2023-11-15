from rq import Worker, Queue, Connection
from redis import Redis
import os


queue_name = os.getenv('QUEUE_NAME', 'test_queue')
redis_client = Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0)
queue = Queue(queue_name, connection=redis_client)


with Connection(redis_client):
    Worker(queue_name, connection=redis_client).work()
    