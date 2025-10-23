### For code please switch to the code branch. The main branch contains presentations and reports

# Proof-of-Learning

This repository is baseed on an implementation of the paper [Proof-of-Learning: Definitions and Practice](https://arxiv.org/abs/2103.05633), published in 42nd IEEE Symposium on
Security and Privacy. This paper, introduces the concept of proof-of-learning in ML. Inspired by research on both proof-of-work and verified computing, the paper observes how a seminal training algorithm, gradient descent, accumulates secret information due to its stochasticity. This produces a natural construction for a proof-of-learning which demonstrates that a party has expended the compute require to obtain a set of model parameters correctly. For more details, please read the paper.

We test our code on two datasets: MNIST, and MedMNIST. 
We also analyse various PoL attack scenarios and experiment with possible defences.

Link to the original repository is here [Original repository by authors](https://github.com/cleverhans-lab/Proof-of-Learning.git)

### Code for our implementation is currently in the code branch.

### Dependency
Our code is implemented and tested on PyTorch. Following packages are used:
```
numpy
pytorch==1.6.0
torchvision==0.7.0
scipy==1.6.0
```

### Train
To train a model and create a proof-of-learning:
```
python train.py --save-freq [checkpointing interval] --dataset [any dataset in torchvision] --model [models defined in model.py or any torchvision model]
```
`save-freq` is checkpointing interval, denoted by k in the paper. There are a few other arguments that you could find at the end of the script. 

Note that the proposed algorithm does not interact with the training process, so it could be applied to any kinds of gradient-descent based models.


### Verify
To verify a given proof-of-learning:
```
python verify.py --model-dir [path/to/the/proof] --dist [distance metric] --q [query budget] --delta [slack parameter]
```
Setting q to 0 or smaller will verify the whole proof, otherwise the top-q iterations for each epoch will be verified. More information about `q` and `delta` can be found in the paper. For `dist`, you could use one or more of `1`, `2`, `inf`, `cos` (if more than one, separate them by space). The first 3 are corresponding l_p norms, while `cos` is cosine distance. Note that if using more than one, the top-q iterations in terms of all distance metrics will be verified.

Please make sure `lr`, `batch-sizr`, `epochs`, `dataset`, `model`, and `save-freq` are consistent with what used in `train.py`.

### Credits to original authors
This project has been derived from the paper:
```
@inproceedings{jia2021proofoflearning,
      title={Proof-of-Learning: Definitions and Practice}, 
      author={Hengrui Jia and Mohammad Yaghini and Christopher A. Choquette-Choo and Natalie Dullerud and Anvith Thudi and Varun Chandrasekaran and Nicolas Papernot},
      booktitle={Proceedings of the 42nd IEEE Symposium on Security and Privacy},
      year={2021}
}
```
