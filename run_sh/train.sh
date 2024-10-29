CUDA_VISIBLE_DEVICES=0 python main.py \
--mode 'train' \
--model 'vit' \
--num_epoch 50 \
--val_on_epoch 1 \
--batch_size 16 \
--lr 0.001 \
--resume 1 \
--task_name 'first train'