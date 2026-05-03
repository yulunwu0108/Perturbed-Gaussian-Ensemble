from typing import List, Dict
import random
from functools import partial
import torch
from copy import deepcopy
from einops import rearrange, reduce, repeat
import math
import itertools


class BaseSchema:
    init_views: List[int]
    load_its: Dict[int, int]

    def __init__(self, **kwargs) -> None:
        self.init_views = []
        self.load_its = {}

    def num_views_to_add(self, it:int) -> int:
        return self.load_its.get(it, 0)


class All(BaseSchema):

    def __init__(self, **kwargs) -> None:
        dataset_size = kwargs.get("dataset_size")
        self.init_views = list(range(dataset_size))
        random.shuffle(self.init_views)
        self.load_its = {}

    def num_views_to_add(self, it: int) -> int:
        return 0


class VNSeqMInplace(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, dataset_size: int, scene, N: int=20, M: int=1, num_init_views: int=4, interval_epochs=100, **kwargs):
        """
        N: int total views to select
        M: # views to select each time
        """
        super().__init__()
        init_interval = 36 // num_init_views
        self.init_views = [i * init_interval for i in range(0, num_init_views)]
        self.num_views_total = N

        self.load_its = {}
        cur_dataset_size = len(self.init_views)
        it_base = cur_dataset_size * interval_epochs
        num_views_left = N - len(self.init_views)

        if num_views_left > 0:
            assert num_views_left % M == 0, "cannot split M evenly to the rest views"

            while num_views_left > 0:
                self.load_its[it_base] = M

                cur_dataset_size += M
                it_base += cur_dataset_size * interval_epochs
                num_views_left -= M


schema_dict: Dict[str, BaseSchema] = {'all': All}


def generate_schema(schema_name: str, dataset_size: int, scene):
    if schema_name in schema_dict.keys():
        return schema_dict[schema_name](dataset_size=dataset_size, scene=scene)
    else:
        target_view_num = int(schema_name.split("v")[1].split("init")[0])
        init_view_num = int(schema_name.split("init")[1].split("_")[0])

        max_interval = 15_000 // ((init_view_num + target_view_num) * (target_view_num - init_view_num + 1) // 2)
        print(f"max interval: {max_interval}")

        schema = partial(VNSeqMInplace, N=target_view_num, M=1, num_init_views=init_view_num, interval_epochs=max_interval)
        return schema(dataset_size=dataset_size, scene=scene)