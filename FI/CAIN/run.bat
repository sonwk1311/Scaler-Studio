@echo off
REM Chạy CAIN training trên Windows

SET CUDA_VISIBLE_DEVICES=0

python main.py ^
    --exp_name CAIN_train ^
    --dataset vimeo90k ^
    --batch_size 8 ^
    --test_batch_size 16 ^
    --model cain ^
    --depth 3 ^
    --loss 1*L1 ^
    --max_epoch 200 ^
    --lr 0.0002 ^
    --log_iter 100 ^
    --num_workers 4
