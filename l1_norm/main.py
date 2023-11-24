import os
import torch
import numpy as np
import requests
import random
import string
from io import BytesIO
from PIL import Image
from l1_norm.prune_l1_norm import GeneratorPruned
from compress import recovery_srgan
from config import DEVICE
from minio_connection import minio_client, bucket_name


def srgan(filename: str, ip_host: str, start_timestamp: str, prune_amount: str):
    input_filepath = f'input_files/{filename}'
    output_filepath = f"output_files/{filename}"

    input_filepath = minio_client.get_object(
        bucket_name=bucket_name,
        object_name=input_filepath,
    )

    try:
        gen = GeneratorPruned(prune_amount=float(prune_amount)/100)
        gen.load_state_dict(torch.load(f'l1_norm/checkpoints/{float(prune_amount)/100}/gen.pth.tar')["state_dict"])
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

        vram_log_path = os.path.splitext(filename)[0] + "".join(random.sample(string.ascii_letters, 3)) + '.txt'
        requests.post(url=f'http://{ip_host}:8000/vram_logs',
            data=dict(filename=vram_log_path,
                        image_filename=filename,
                        start_timestamp=start_timestamp,
                        tipe_model=f"l1_norm_{prune_amount}",
                        ip_host_manager=os.getenv('MANAGER_HOST', 'localhost'),
                    )
            )

        print(output_filepath)    
        # print(image)
    except RuntimeError as e:
        requests.post(url=f'http://{ip_host}:8000/vram_logs',
            data=dict(filename='-',
                    image_filename=filename,
                    start_timestamp=start_timestamp,
                    tipe_model=f"l1_norm_{prune_amount}",
                    ip_host_manager=os.getenv('MANAGER_HOST', 'localhost'),
                    status_process='Fail', message_process=e.__str__(),
                )
        )


if __name__ == "__main__":
    srgan(
        filename=os.getenv("FILENAME", ''),
        ip_host=os.getenv("IP_HOST", "localhost"),
        start_timestamp=os.getenv('START_TIMESTAMP', ''),
        prune_amount = os.getenv('PRUNE_AMOUNT', '0'),
    )
    