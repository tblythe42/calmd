"""Module containing executor classes for running model repetitions.


"""
from collections.abc import Iterator

from tqdm import tqdm

class Sequential:

    def __init__(self, process):
        self.process = process

    def __call__(self, jobs: list | Iterator) -> Iterator:
        for job in tqdm(jobs, desc='running simulations', leave=False):
            yield self.process(job)


class MultiThread:

    def __init__(self, process):
        self.process = process

    def __call__(self, jobs: list | Iterator):
        return None