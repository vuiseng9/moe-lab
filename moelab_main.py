#!/usr/bin/env python

import os
import sys
import logging

import datasets
import transformers
from transformers import HfArgumentParser, Trainer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from neoclm import ModelArgs, DataArgs, OpinionatedTrainArgs
from neoclm import clm, clm_sweep_lr
from moelab import MOELAB_TRAINER_CLS

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0")

logger = logging.getLogger(__name__)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArgs, DataArgs, OpinionatedTrainArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Set log level for neoclm module logger
    logging.getLogger("neoclm").setLevel(log_level)

    TrainerCls = Trainer
    if model_args.model_type in MOELAB_TRAINER_CLS:
        TrainerCls = MOELAB_TRAINER_CLS[model_args.model_type]

    if training_args.sweep_lr:
        clm_sweep_lr(model_args, data_args, training_args, TrainerCls)
    else:
        clm(model_args, data_args, training_args, TrainerCls)


if __name__ == "__main__":
    main()
