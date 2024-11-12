import os
import pandas as pd
import matplotlib.pyplot as plt

block = [32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]
size = [2 ** i for i in range (1, 20)]

block_matr = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
size_matr = [2 ** i for i in range (1, 10)]

for b in block:
    for s in size:
        os.system("CUDA_VISIBLE_DEVICES=2 ../build/01-add {} {}".format(s, b))
        os.system("CUDA_VISIBLE_DEVICES=2 ../build/02-mul {} {}".format(s, b))
        os.system("CUDA_VISIBLE_DEVICES=2 ../build/05-scalar-mul {} {}".format(s, b))
        os.system("CUDA_VISIBLE_DEVICES=2 ../build/06-cosine-vector {} {}".format(s, b))

for b in block:
    for s in matr_size:
        os.system("CUDA_VISIBLE_DEVICES=2 ../build/03-matrix-add {} {} {}".format(s, s, b))
        os.system("CUDA_VISIBLE_DEVICES=2 ../build/04-matrix-vector-mul {} {} {}".format(s, s, b))
        os.system("CUDA_VISIBLE_DEVICES=2 ../build/07-matrix-mul {} {} {}".format(s, s, b))
