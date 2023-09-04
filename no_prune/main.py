from io import BytesIO
import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image
from model import Generator
from compress import recovery_srgan
from config import DEVICE
from minio_connection import minio_client, bucket_name

if __name__ == "__main__":
    input_filepath = f'input_files/{os.getenv("FILENAME")}'
    output_filepath = f"output_files/{os.getenv('FILENAME')}"
    
    input_filepath = minio_client.get_object(
        bucket_name=bucket_name,
        object_name=input_filepath,
    )
    
    gen = Generator()
    gen.load_state_dict(torch.load('gen.pth.tar')["state_dict"])
    gen.eval().to(DEVICE)

    input_image = np.asarray(Image.open(input_filepath))
    # input_image = compress_binary(input_image)
    # input_image = recovery_binary(input_image)
    input_image = recovery_srgan(img=input_image, gen=gen)
    output_bytes = BytesIO()
    Image.fromarray(input_image).save(output_bytes, format='PNG')
    output_bytes = output_bytes.getvalue()
    minio_client.put_object(
        bucket_name=bucket_name,
        object_name=output_filepath,
        data=BytesIO(output_bytes),
        length=len(output_bytes),
    )

    print(output_filepath)    
    # print(image)
