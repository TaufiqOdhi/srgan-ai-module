from rq import Worker, Queue, Connection
from redis import Redis


queue_name = 'test_queue'
redis_client = Redis(host='localhost', port=6379, db=0)
queue = Queue(queue_name, connection=redis_client)


with Connection(redis_client):
    Worker(queue_name, connection=redis_client).work()
    