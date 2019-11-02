""" 
Description of the experiment and objectives
"""
from data import transform
from experiment import Experiment

class TestExperiment(Experiment):
    raw_data ="train_data_small.csv"
    transforms = [
        transform.t0_test
    ]

if __name__ == '__main__':
    te = TestExperiment("test")
    te.run()