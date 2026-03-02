# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from cfm_mppi.models.ema import EMA
from cfm_mppi.models.transformer import TransformerModel

MODEL_CONFIGS = {
    "transformer": {
        "in_channels": 2,
        "out_channels": 2,
        "d_model": 256,
        "nhead": 4,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "max_len": 500,
    },
}   


def instantiate_model(
    architechture: str, is_discrete: bool, use_ema: bool
):
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    model = TransformerModel(**MODEL_CONFIGS[architechture])

    if use_ema:
        return EMA(model=model)
    else:
        return model
