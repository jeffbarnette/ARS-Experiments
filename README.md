# Augmented Random Search (ARS) Experiments

A new type of reinforcement learning approach for artificial intelligence and machine learning using simple random search methods. The details of this research was published in a recent paper by Horia Mania, Aurelia Guy and Benjamin Recht in the Department of Electrical Engineering and Computer Science at the University of California, Berkeley. The paper is titled "Simple random search provides a competitive approach to reinforcement learning."

I decided to experiment with their Augmented Random Search V2 Algorithm using OpenAI Gym and the Half Cheetah environment to see what kind of results I could get using Python and some of the standard Python libraries. As I experiment with this further, I will continue to update this repository with additional experiments.

## Background:

“A common belief in model-free reinforcement learning is that methods based on random search in the parameter space of policies exhibit signiﬁcantly worse sample complexity than those that explore the space of actions. The authors of the paper dispel such beliefs by introducing a random search method for training static, linear policies for continuous control problems, matching state-of-the-art sample eﬃciency on the benchmark MuJoCo locomotion tasks. Their method also ﬁnds a nearly optimal controller for a challenging instance of the Linear Quadratic Regulator, a classical problem in control theory, when the dynamics are not known. Computationally, their random search algorithm is at least 15 times more eﬃcient than the fastest competing model-free methods on these benchmarks. They take advantage of this computational eﬃciency to evaluate the performance of their method over hundreds of random seeds and many diﬀerent hyperparameter conﬁgurations for each benchmark task. Their simulations highlight a high variability in performance in these benchmark tasks, suggesting that commonly used estimations of sample eﬃciency do not adequately evaluate the performance of reinforcement learning algorithms.”

## Features:

* Written in Python 3.5.x
* Uses Open AI Gym

## Running the app

### Install Dependencies

Use `pip` to install the app's dependencies.

    pip install gym
    pip install pybullet

Also required is `ffmpeg`. On a Mac, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.

You will also need to make sure you are using Python 3.8.x or later.

### Run the app

    python ars.py

This process will take quite some time depending on the specs and performance of your computer. Expect to let it run for several hours while the model teaches itself to walk in one direction towards its reward (right edge of the screen). Small video clips of each attempt after training has occurred will be stored in the 'exp/brs/monitor' folder which you can view as the files are saved. It may take a few hundred steps to be completed before you will see any readable output in the monitor.

### Further Experimentation

You can run this code using other Open AI Gym models to see what kind of results you get.
