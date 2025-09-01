import subprocess

# for i in range(5):
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k1 --batch-size 128 --save-freq 1', shell=True)
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k2 --batch-size 128 --save-freq 2', shell=True)
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k5 --batch-size 128 --save-freq 5', shell=True)
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k10 --batch-size 128 --save-freq 10', shell=True)
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k20 --batch-size 128 --save-freq 20', shell=True)
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k50 --batch-size 128 --save-freq 50', shell=True)
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k100 --batch-size 128 --save-freq 100', shell=True)
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k200 --batch-size 128 --save-freq 200', shell=True)
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k300 --batch-size 128 --save-freq 300', shell=True)
#     subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_k400 --batch-size 128 --save-freq 400', shell=True)


for i in range(5):
    subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_a1e-3 --batch-size 128 --save-freq 20 --lr 0.001', shell=True)
    subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_a5e-3 --batch-size 128 --save-freq 20 --lr 0.005', shell=True)
    subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_a1e-2 --batch-size 128 --save-freq 20 --lr 0.01', shell=True)
    subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_a5e-2 --batch-size 128 --save-freq 20 --lr 0.05', shell=True)
    subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_a1e-1 --batch-size 128 --save-freq 20 --lr 0.1', shell=True)
    subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_a5e-1 --batch-size 128 --save-freq 20 --lr 0.5', shell=True)
    subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_a1e0 --batch-size 128 --save-freq 20 --lr 1', shell=True)
    subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_a5e0 --batch-size 128 --save-freq 20 --lr 5', shell=True)
    subprocess.run(f'python train.py --dataset MNIST --model resnet20 --epochs 10 --id {i+1}_b128_a1e1 --batch-size 128 --save-freq 20 --lr 10', shell=True)
