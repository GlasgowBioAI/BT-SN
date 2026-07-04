"""
BT-SN hyperparameter configuration.

The main config dict is shared across training scripts.
Dataset-specific paths are selected automatically based on `testing_dataset_name`.
"""
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    # ── Reproducibility ──
    'seed': 42,

    # ── Training ──
    'batch_size': [256],
    'learning_rate': [1e-3],
    'num_epochs': 1500,
    'optimizer': ['Adam'],
    'scheduler': 'ReduceLROnPlateau',
    'lr_decay_factor': 0.1,
    'lr_patience': 100,
    'min_lr': 1e-6,

    # ── Encoder architecture ──
    'encoder_first_layer_dim': [1024],
    'encoder_num_layers': [3],
    'encoder_output_dim': [4],

    # ── Structural noise augmentation ──
    'lambda_pseudo_bulk_rate1': [0.15, 0.85],      # alpha range for view 1
    'lambda_pseudo_bulk_rate2': [0.15, 0.85],      # alpha range for view 2

    # ── Barlow Twins loss ──                # off-diagonal penalty
    'use_projector': [True],

    # ── Regularisation ──
    'regularization': ['l1', 'l2', 'elastic'],
    'l1_regularization_lambda': 1e-4,
    'l2_regularization_lambda': 1e-3,

    # ── Model type ──
    'model_type': 'barlow_twins',                   # 'barlow_twins' | 'auto_encoder' | 'VAE' | 'scarf' | 'vime' | 'byol'

    # ── Data ──
    'testing_dataset_name': 'TCGA',                 # 'TCGA' | 'Metabric' | 'scanb'
    'test_size': 0.2,                               # train/test split ratio
    'distortion_add': 'bulk',                       # 'bulk' (structural noise) | 'sc' (single-cell)
    'sample_size': 500,                             # cells/resamples per pseudo-bulk (sc mode)

    # ── Scaling ──

    # ── Paths (base directory: project root) ──
    'bulk_path': '',
    'bulk_label_path': '',
    'normal_bulk_path': '',
    'survival_path': '',
    'bulk_id_path': '',
    'encoder_file': '',
    'projector_file': '',
    'decoder_file': '',
    'model_save_path': './model_checkpoints',

    # ── Device ──
    'device': device,
}

