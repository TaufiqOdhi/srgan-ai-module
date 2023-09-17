from minio import Minio
import os


#Initialize Minio Client
minio_client = Minio(
    endpoint=f"{os.getenv('MINIO_HOST', 'localhost')}:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False, # disable SSL
)

bucket_name = "super-resolution"
