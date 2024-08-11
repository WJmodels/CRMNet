# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MyDatasetISIC2018Task1(BaseSegDataset):
    """MyDataset.

    """

    # 改成自己的类别，并设置对应数量的颜色
    METAINFO = dict(
        classes=('background', 'Lesion'),
        palette=[[0, 0, 0], [128, 0, 0],])


    def __init__(self,
                 seg_map_suffix='.png',
                 reduce_zero_label=False, 
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

