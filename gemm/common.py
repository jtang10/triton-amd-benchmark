import abc
from dataclasses import dataclass



@dataclass()
class TritonParameters:
    num_warps: int