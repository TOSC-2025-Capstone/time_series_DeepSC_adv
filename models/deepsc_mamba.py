import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

from omegaconf import OmegaConf

import sys
import os

from samba_mixer.model.input_projections.linear_projection_time_embedding_cycle_diff_embedding import LinearProjectionWithLocalTimeAndGlobalDiffEmbedding
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/d:/chanminLee/time_series_DeepSC_adv')

# input_projection=SambaInputProjectionFactory.get_input_projection(model_config),
LinearProjectionWithLocalTimeAndGlobalDiffEmbedding(d_model)