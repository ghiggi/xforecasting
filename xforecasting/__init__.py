#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:06:20 2022

@author: ghiggi
"""
# TODO: use AR acronym for all 
# file naming: dataloader_autoregressive --> dataloader 
# file naming: training_autoregressive --> training 
# file naming: predictions_autoregressive --> predictions 

from .scheduler import AR_Scheduler
from .dataloader_autoregressive import AutoregressiveDataset, AutoregressiveDataLoader
from .training_autoregressive import AutoregressiveTraining
from .predictions_autoregressive import AutoregressivePredictions, rechunk_forecasts_for_verification
from .training_info import AR_TrainingInfo
from .early_stopping import EarlyStopping 

from xforecasting import (
    AR_Scheduler,
    AR_TrainingInfo,
    AutoregressiveDataset,
    AutoregressiveDataLoader,
    AutoregressiveTraining,
    AutoregressivePredictions,
    rechunk_forecasts_for_verification,
    EarlyStopping,
)

__all__ = [
    "AR_Scheduler",
    "AR_TrainingInfo",
    "AutoregressiveDataset",
    "AutoregressiveDataLoader",
    "AutoregressiveTraining",
    "AutoregressivePredictions",
    "rechunk_forecasts_for_verification",
    "EarlyStopping",

]
