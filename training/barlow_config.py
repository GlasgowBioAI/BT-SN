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
    'seeds': [42, 43, 44],

    # ── Training ──
    'batch_size': 256,
    'learning_rate': 1e-3,
    'num_epochs': 1500,
    'optimizer': ['Adam'],
    'scheduler': 'ReduceLROnPlateau',
    'lr_decay_factor': 0.1,
    'lr_patience': 100,
    'min_lr': 1e-6,

    # ── Encoder architecture ──
    'encoder_hidden_dims': [1024, 512, 256],
    'encoder_output_dim': 4,

    # ── Structural noise augmentation ──
    # alpha is the tumour coefficient. It is sampled independently for every
    # sample and view at every epoch; the normal coefficient is (1 - alpha).
    'alpha_min': 0.15,
    'alpha_max': 0.85,

    # ── Barlow Twins loss ──                # off-diagonal penalty
    'use_projector': True,
    'projector_hidden_dim': 128,
    'projector_output_dim': 256,
    'lambda_param': 18.30605412,

    # ── Regularisation ──
    'regularization': 'elastic',
    'l1_regularization_lambda': 2.18e-6,
    'l2_regularization_lambda': 4.50e-5,

    # ── Model type ──
    'model_type': 'barlow_twins',                   # 'barlow_twins' | 'auto_encoder' | 'VAE' | 'scarf' | 'vime' | 'byol'

    # ── Data ──
    'testing_dataset_name': 'TCGA',                 # 'TCGA' | 'Metabric' | 'scanb'
    'test_size': 0.2,
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

