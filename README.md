# Vision Transformer for MNIST classification
Self-implemented Vision Transformer model for classification of handwritten digits.
## Train 
The training script is in the `run_sh` folder, you can run:
```bash
bash run_sh/train.sh
```
## Test
The test script is in the `run_sh` folder, you can run:
```bash
bash run_sh/test.sh
```
## Tensorboard
You can use tensorboard to monitor the training process:
```bash 
tensorboard --logdir log
```