# Acknowledgement

This is a fork of the  [Junshan-Wang/ContinualGNN](https://github.com/Junshan-Wang/ContinualGNN) repository.
Changes can be seen as the latest commits.
Example usage:

`python3.7 my_main_stream.py --cuda --gnn_type=gat --heads=16 --sampler=rand --new_ratio=0.2`


# Streaming Graph Neural Networks via Continual Learning

Code for [Streaming Graph Neural Networks via Continual Learning](https://dl.acm.org/doi/abs/10.1145/3340531.3411963)（CIKM 2020). ContinualGNN is a streaming graph neural network based on continual learning so that the model is trained incrementally and up-to-date node representations can be obtained at each time step.

### Requirements

* python = 3.8.5
* pytorch = 1.7.1
* scikit-learn = 0.23.2

### Usages

* ContinualGNN (proposed model) on Cora:
```
cd src/
python main_stream.py --data=cora --new_ratio=0.8 --memory_size=250 --ewc_lambda=80.0 
```
* OnlineGNN (lower bound) on Cora:
```
python main_stream.py --data=cora
```

If using cuda, set `--cuda`.
