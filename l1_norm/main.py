import os
import torch
import numpy as np
import requests
import random
import string
from io import BytesIO
from PIL import Image
from prune_l1_norm import GeneratorPruned
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
    
    gen = GeneratorPruned()
    gen.load_state_dict(torch.load(f'checkpoints/{os.getenv("PRUNE_AMOUNT")}/gen.pth.tar')["state_dict"])
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

    vram_log_path = os.path.splitext(os.getenv('FILENAME'))[0] + "".join(random.sample(string.ascii_letters, 3)) + '.txt'
    requests.post(url=f'http://{os.getenv("IP_HOST", "localhost")}:8000/vram_logs',
                  data=dict(filename=vram_log_path,
                            image_filename=os.getenv("FILENAME"),
                            start_timestamp=os.getenv('START_TIMESTAMP', ''),
                            tipe_model=f"li_norm_{os.getenv('PRUNE_AMOUNT', '')}"
                        )
                )

    print(output_filepath)    
    # print(image)
