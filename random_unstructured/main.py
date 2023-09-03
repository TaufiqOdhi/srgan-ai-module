import os
import torch
import numpy as np
from PIL import Image
from prune_random_unstructured_global import GeneratorPruned
from compress import recovery_srgan
from config import DEVICE


if __name__ == "__main__":
    input_filepath = f'input_files/{os.getenv("FILENAME")}'
    output_filepath = f"output_files/{os.getenv('FILENAME')}"
    
    gen = GeneratorPruned()
    gen.load_state_dict(torch.load(f'checkpoints/{os.getenv("PRUNE_AMOUNT")}/gen.pth.tar')["state_dict"])
    gen.eval().to(DEVICE)

    input_image = np.asarray(Image.open(input_filepath))
    # input_image = compress_binary(input_image)
    # input_image = recovery_binary(input_image)
    input_image = recovery_srgan(img=input_image, gen=gen)
    Image.fromarray(input_image).save(output_filepath)

    print(output_filepath)    
    # print(image)
