from minio import Minio


#Initialize Minio Client
minio_client = Minio(
    endpoint="localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False, # disable SSL
)

bucket_name = "super-resolution"
