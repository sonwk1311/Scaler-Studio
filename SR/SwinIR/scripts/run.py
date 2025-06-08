import os

# Prepare dataset
os.system("python ./scripts/prepare_dataset.py --images_dir C:/Users/Administrator/Documents/SwinIR/datasets/DFO2K/share/jywang/dataset/df2k_ost/GT --output_dir ../datasets/DFO2K --image_size 544 --step 272 --num_workers 20")
