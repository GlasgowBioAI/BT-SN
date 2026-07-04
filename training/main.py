from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, confusion_matrix, \
    classification_report
from barlow_config import *
from evaluation import *
from training import *
from dataset_input import *
from model_structure import *
from lifelines.exceptions import ConvergenceError
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()



def setup_logging(result_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(result_dir, f"training_{timestamp}.log")

    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])
    return log_file


def log_config_and_model(config, models):
    logging.info("Configuration Parameters:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")

    logging.info("Model Architectures:")
    for model_name, model in models.items():
        logging.info(f"{model_name}: {model}")


def train_and_evaluate():
    try:
        #  
        if config['model_type'] == 'barlow_twins':
            config.update(BEST_PARAMS_BT)
        elif config['model_type'] == 'auto_encoder':
            config.update(BEST_PARAMS_AE)
        elif config['model_type'] == 'VAE':
            config.update(BEST_PARAMS_VAE)
        elif config['model_type'] == 'scarf':
            config.update(BEST_PARAMS_SCARF)
        elif config['model_type'] == 'vime':
            config.update(BEST_PARAMS_VIME)
        elif config['model_type'] == 'byol':
            config.update(BEST_PARAMS_BYOL)

        #  
        testing_dataset_name = config['testing_dataset_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join("..", "result", f"{config['model_type']}_{testing_dataset_name}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        #  
        log_file = setup_logging(result_dir)
        logging.info(f"Log file saved to: {log_file}")

        #  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        #  
        set_seed(42)

        #  
        logging.info("Stage 1: Loading data...")
        if config['testing_dataset_name'] == 'scanb':
            (train_dataloader, eval_dataloader, eval_bulk,
             eval_bulk_labels, eval_survival_df, distortions_list, normal_bulk) = scanb_Barlow_dataloader(
                config['sc_path'],
                config['bulk_path'],
                config['bulk_id_path'],
                config['normal_bulk_path'],
                config['batch_size'],
                config['lambda_pseudo_bulk_rate1'],
                config['lambda_pseudo_bulk_rate2'],
            )

        else:
            (train_dataloader, eval_dataloader, eval_bulk,
             eval_bulk_labels, eval_survival_df, distortions_list, normal_bulk) = Barlow_dataloader(
                config['sc_path'],
                config['bulk_path'],
                config['bulk_id_path'],
                config['survival_path'],
                config['normal_bulk_path'],
                config['batch_size'],
                config['lambda_pseudo_bulk_rate1'],
                config['lambda_pseudo_bulk_rate2'],
                config['test_size']
            )
        input_dim = eval_bulk.shape[1]

        #  
        logging.info("Stage 2: Initializing models...")


        encoder_first_layer_dim = config['encoder_first_layer_dim'][0]
        hidden_dims = [encoder_first_layer_dim]
        encoder_num_layers = config['encoder_num_layers'][0]
        encoder_output_dim = config['encoder_output_dim'][0]
        lambda_param = config['lambda_param']
        device = config['device']
        regularization = config['regularization']
        l1_regularization_lambda = config['l1_regularization_lambda']
        l2_regularization_lambda = config['l2_regularization_lambda']
        beta_vae = config['beta_vae']
        lambda_pseudo_bulk_rate1 = config['lambda_pseudo_bulk_rate1']
        lambda_pseudo_bulk_rate2 = config['lambda_pseudo_bulk_rate2']
        encoder = None
        projector = None
        decoder = None
        num_epochs = config['num_epochs']
        optimizer = config['optimizer'][0]
        learning_rate = config['learning_rate'][0]


        for i in range(1, encoder_num_layers):
            next_dim = hidden_dims[-1] // 2
            if next_dim < encoder_output_dim:  #  
                break
            hidden_dims.append(next_dim)

        encoder = Encoder(input_dim, hidden_dims, encoder_output_dim).to(device)

        if config['model_type'] == 'auto_encoder':
            decoder = Decoder(encoder_output_dim, hidden_dims, input_dim).to(device)
            criterion = nn.MSELoss()
            params = list(encoder.parameters()) + list(decoder.parameters())

        elif config['model_type'] == 'barlow_twins':
            _up = config.get('use_projector', True)
            if isinstance(_up, list):
                use_projector = bool(_up[0])
            else:
                use_projector = bool(_up)
            if use_projector:
                #   Decoder  ：Decoder(encoder_output_dim, hidden_dims, input_dim)
                projector_out_dim = input_dim
                projector = Projector(encoder_output_dim, hidden_dims[::-1], projector_out_dim).to(device)
                params = list(encoder.parameters()) + list(projector.parameters())
            else:
                projector = None
                params = list(encoder.parameters())
            #   BatchNorm  ：
            # -   projector   projector  （  input_dim）
            # -   projector   encoder_output_dim
            criterion_output_dim = projector_out_dim if use_projector else encoder_output_dim
            criterion = BarlowTwinsLoss(lambda_param, criterion_output_dim).to(device)


        elif config['model_type'] == 'VAE':
            encoder = VAE_Encoder(input_dim, hidden_dims, encoder_output_dim).to(device)
            decoder = Decoder(encoder_output_dim, hidden_dims, input_dim).to(device)
            criterion = nn.MSELoss()
            params = list(encoder.parameters()) + list(decoder.parameters())

        elif config['model_type'] == 'scarf':
            # SCARF: simple 3-layer MLP projector: 4 → 64 → 256
            projector = nn.Sequential(
                nn.Linear(encoder_output_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 256),
            ).to(device)
            criterion = InfoNCELoss(temperature=config.get('temperature', 0.5)).to(device)
            params = list(encoder.parameters()) + list(projector.parameters())

        elif config['model_type'] == 'vime':
            vime_decoder = VIMEDecoder(encoder_output_dim, hidden_dims, input_dim).to(device)
            criterion = VIMELoss(alpha=config.get('alpha', 1.0)).to(device)
            params = list(encoder.parameters()) + list(vime_decoder.parameters())

        elif config['model_type'] == 'byol':
            encoder_online = encoder
            encoder_target = Encoder(input_dim, hidden_dims, encoder_output_dim).to(device)
            encoder_target.load_state_dict(encoder_online.state_dict())

            projector_online = Projector(encoder_output_dim, hidden_dims[::-1], input_dim).to(device)
            projector_target = Projector(encoder_output_dim, hidden_dims[::-1], input_dim).to(device)
            projector_target.load_state_dict(projector_online.state_dict())

            predictor = BYOLPredictor(input_dim, hidden_dims[::-1], input_dim).to(device)
            criterion = None
            params = (list(encoder_online.parameters()) +
                      list(projector_online.parameters()) +
                      list(predictor.parameters()))
            config['encoder_target'] = encoder_target
            config['projector_target'] = projector_target
            config['predictor'] = predictor

        else:
            raise ValueError(f"Unsupported model_type: {config['model_type']}")

        # Inference encoder: for BYOL, use encoder_online; others use 'encoder'
        inference_encoder = encoder_online if config['model_type'] == 'byol' else encoder

        optimizer_class = getattr(optim, optimizer)
        optimizer = optimizer_class(
            params,
            lr= learning_rate,
            **config.get('optimizer_params', {})
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['lr_decay_factor'],
            patience=config['lr_patience'],
            min_lr=config['min_lr'],
            verbose=True
        )

        #  （ ）
        #  
        models = {"Encoder": encoder if config['model_type'] != 'byol' else encoder_online}
        if config['model_type'] == 'auto_encoder' or config['model_type'] == 'VAE':
            models["Decoder"] = decoder
        elif config['model_type'] == 'barlow_twins' or config['model_type'] == 'scarf':
            if projector is not None:
                models["Projector"] = projector
        elif config['model_type'] == 'byol':
            models["Projector_Online"] = projector_online
            models["Projector_Target"] = projector_target
            models["Predictor"] = predictor
        elif config['model_type'] == 'vime':
            models["VIMEDecoder"] = vime_decoder
        log_config_and_model(config, models)

        #  
        min_loss = float('inf')
        best_epoch = -1
        losses = []
        embedding_df = None

        logging.info("Stage 3: Training...")
        for epoch in range(num_epochs):
            old_lr = optimizer.param_groups[0]['lr']

            if config['model_type'] == 'auto_encoder':
                epoch_loss = train_auto_encoder(encoder, decoder, train_dataloader, criterion, optimizer, scaler,
                                                        device,
                                                        normal_bulk,
                                                        lambda_pseudo_bulk_rate1,
                                                        lambda_pseudo_bulk_rate2,
                                                        regularization,
                                                        l1_regularization_lambda,
                                                        l2_regularization_lambda,
                                                        )
                final_c = None
            elif config['model_type'] == 'VAE':
                epoch_loss = train_vae(encoder, decoder, train_dataloader, criterion, optimizer, scaler,
                                                        device, beta_vae)
                final_c = None
            elif config['model_type'] == 'scarf':
                epoch_loss, final_c = train_scarf(
                    encoder, projector, train_dataloader, criterion, optimizer, scaler,
                    device, normal_bulk,
                    lambda_pseudo_bulk_rate1, lambda_pseudo_bulk_rate2,
                    regularization, l1_regularization_lambda, l2_regularization_lambda
                )
            elif config['model_type'] == 'vime':
                epoch_loss, _, _ = train_vime(
                    encoder, vime_decoder, train_dataloader, criterion, optimizer, scaler,
                    device,
                    mask_prob=config.get('mask_prob', 0.3),
                    alpha=config.get('alpha', 1.0)
                )
                final_c = None
            elif config['model_type'] == 'byol':
                epoch_loss, final_c = train_byol(
                    encoder_online, encoder_target,
                    projector_online, projector_target,
                    predictor,
                    train_dataloader, optimizer, scaler, device, normal_bulk,
                    lambda_pseudo_bulk_rate1, lambda_pseudo_bulk_rate2,
                    tau=config.get('tau', 0.996),
                    regularization=regularization,
                    l1_regular_lambda=l1_regularization_lambda,
                    l2_regular_lambda=l2_regularization_lambda
                )
            else:
                epoch_loss, last_c = train_barlow_twins(encoder, projector, train_dataloader, criterion, optimizer, scaler,
                                                device,
                                                normal_bulk,
                                                lambda_pseudo_bulk_rate1,
                                                lambda_pseudo_bulk_rate2,
                                                regularization,
                                                l1_regularization_lambda,
                                                l2_regularization_lambda,
                                                )
                final_c = last_c

            losses.append(epoch_loss)
            scheduler.step(epoch_loss)

            new_lr = optimizer.param_groups[0]['lr']

            #   lr  ，  logger  
            if new_lr != old_lr:
                logging.info(
                    f"LR from {old_lr:.1e} to {new_lr:.1e}"
                )
            print(f"loss: {epoch_loss:.4f}")

            if (epoch + 1) % 100 == 0:
                #  
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}, Learning Rate: {current_lr:.1e}")

            #  
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                best_epoch = epoch + 1
                distortion_embeddings_list = []

                inference_encoder.eval()
                with torch.no_grad():
                    #   DataLoader（shuffle=False） ， 
                    #  ：eval_dataloader   DynamicBarlowDataset(mode='eval')   log2(x+1)
                    #   bulk_batch  ，  encoder。
                    embeddings_list = []
                    for bulk_batch, indices in eval_dataloader:
                        bulk_batch = bulk_batch.to(device)
                        if config['model_type'] == 'VAE':
                             _, embedding_batch, _ = inference_encoder(bulk_batch)
                        else:
                            embedding_batch = inference_encoder(bulk_batch)
                        embeddings_list.append(embedding_batch.cpu())
                    best_embedding = torch.cat(embeddings_list, dim=0)

                    for distortion_tensor in distortions_list:
                        embeddings = []
                        #   batch   embedding
                        for i in range(0, distortion_tensor.shape[0], config['batch_size']):
                            batch = distortion_tensor[i:i + config['batch_size']].to(device)
                            if config['model_type'] == 'VAE':
                                 _, embedding_batch, _ = inference_encoder(batch)
                            else:
                                embedding_batch = inference_encoder(batch)
                            embeddings.append(embedding_batch.cpu())
                        #   distortion_tensor   batch   tensor
                        distortion_embedding = torch.cat(embeddings, dim=0)
                        distortion_embedding_df = pd.DataFrame(distortion_embedding.numpy(), index=eval_bulk.index)
                        distortion_embeddings_list.append(distortion_embedding_df)

                #   bulk_tensor_index  （  eval_dataloader  ）
                embedding_df = pd.DataFrame(best_embedding.numpy(), index=eval_bulk.index)

                #  
                embedding_file = os.path.join(result_dir, "embedding.pt")
                torch.save(embedding_df, embedding_file)


                corr_matrix = embedding_df.corr(method='pearson')  # shape: [embedding_dim, embedding_dim]
                corr_file = os.path.join(result_dir, "embedding_self_correlation.csv")
                corr_matrix.to_csv(corr_file)

                #   numpy array
                C = corr_matrix.values
                dim = C.shape[0]
                diag_deviation = np.abs(np.diag(C) - 1).mean()
                off_diag_mask = ~np.eye(dim, dtype=bool)
                off_diag_deviation = np.abs(C[off_diag_mask]).mean()
                correlation_deviation_score = diag_deviation + off_diag_deviation


                #  
                loss_curve_file = os.path.join(result_dir, "loss_curve.png")
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, epoch + 2), losses, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Current Loss Curve')
                plt.legend()
                plt.grid(True)
                plt.savefig(loss_curve_file)
                plt.close()

                #  
                saved_encoder = inference_encoder if config.get("model_type") == "byol" else encoder
                encoder_file = os.path.join(result_dir, "encoder.pt")
                torch.save(saved_encoder.state_dict(), encoder_file)

                if config['model_type'] == 'auto_encoder' or config['model_type'] == 'VAE':
                    decoder_file = os.path.join(result_dir, "decoder.pt")
                    torch.save(decoder.state_dict(), decoder_file)
                else:
                    if projector is not None:
                        projector_file = os.path.join(result_dir, "projector.pt")
                        torch.save(projector.state_dict(), projector_file)

                logging.info(f"Epoch {best_epoch} with loss: {min_loss:.4f}")

        
        print("Correlation matrix (partial view):")
        print(pd.DataFrame(C).iloc[:5, :5])
        
        #  
        print(f"\nDiagonal deviation: {diag_deviation:.4f}")
        print(f"Off-diagonal deviation: {off_diag_deviation:.4f}")
        print(f"Correlation deviation score: {correlation_deviation_score:.4f}")
        if final_c is not None:
            torch.save(final_c, os.path.join(result_dir, "final_c.pt"))  #  
            logging.info(f"Final correlation matrix shape: {final_c.shape}")
            logging.info(f"Final correlation matrix sample:\n{final_c}")

        #  
        logging.info("Stage 4: Evaluation...")

        result = evaluate_clustering_method(embedding_df,
                                            eval_bulk_labels, eval_survival_df, encoder_output_dim,
                                            distortion_embeddings_list, correlation_deviation_score)

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


def train_and_evaluate_with_gaussian_noise():
    """
     Gaussian distortion 。
     train_and_evaluate() ， Gaussian distortion，
     adjacent normal distortion。
    """
    try:
        #  
        if config['model_type'] == 'barlow_twins':
            config.update(BEST_PARAMS_BT)  #  
        elif config['model_type'] == 'auto_encoder':
            config.update(BEST_PARAMS_AE)  #  
        elif config['model_type'] == 'VAE':
            config.update(BEST_PARAMS_VAE)  #  

        #  
        testing_dataset_name = config['testing_dataset_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join("..", "result", f"{config['model_type']}_{testing_dataset_name}_gaussian_noise_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        #  
        log_file = setup_logging(result_dir)
        logging.info(f"Log file saved to: {log_file}")
        logging.info("Using Gaussian noise distortion instead of adjacent normal")

        #  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        #  
        set_seed(42)

        #  
        logging.info("Stage 1: Loading data...")
        if config['testing_dataset_name'] == 'scanb':
            (train_dataloader, eval_dataloader, eval_bulk,
             eval_bulk_labels, eval_survival_df, distortions_list, normal_bulk) = scanb_Barlow_dataloader(
                config['sc_path'],
                config['bulk_path'],
                config['bulk_id_path'],
                config['normal_bulk_path'],
                config['batch_size'],
                config['lambda_pseudo_bulk_rate1'],
                config['lambda_pseudo_bulk_rate2'],
            )

        else:
            (train_dataloader, eval_dataloader, eval_bulk,
             eval_bulk_labels, eval_survival_df, distortions_list, normal_bulk) = Barlow_dataloader(
                config['sc_path'],
                config['bulk_path'],
                config['bulk_id_path'],
                config['survival_path'],
                config['normal_bulk_path'],
                config['batch_size'],
                config['lambda_pseudo_bulk_rate1'],
                config['lambda_pseudo_bulk_rate2'],
                config['test_size']
            )
        input_dim = eval_bulk.shape[1]

        #  
        logging.info("Stage 2: Initializing models...")

        encoder_first_layer_dim = config['encoder_first_layer_dim'][0]
        hidden_dims = [encoder_first_layer_dim]
        encoder_num_layers = config['encoder_num_layers'][0]
        encoder_output_dim = config['encoder_output_dim'][0]
        lambda_param = config['lambda_param']
        device = config['device']
        regularization = config['regularization']
        l1_regularization_lambda = config['l1_regularization_lambda']
        l2_regularization_lambda = config['l2_regularization_lambda']
        beta_vae = config['beta_vae']
        lambda_pseudo_bulk_rate1 = config['lambda_pseudo_bulk_rate1']
        lambda_pseudo_bulk_rate2 = config['lambda_pseudo_bulk_rate2']
        encoder = None
        projector = None
        decoder = None
        num_epochs = config['num_epochs']
        optimizer = config['optimizer'][0]
        learning_rate = config['learning_rate'][0]

        for i in range(1, encoder_num_layers):
            next_dim = hidden_dims[-1] // 2
            if next_dim < encoder_output_dim:  #  
                break
            hidden_dims.append(next_dim)

        encoder = Encoder(input_dim, hidden_dims, encoder_output_dim).to(device)

        if config['model_type'] == 'auto_encoder':
            decoder = Decoder(encoder_output_dim, hidden_dims, input_dim).to(device)
            criterion = nn.MSELoss()
            params = list(encoder.parameters()) + list(decoder.parameters())

        elif config['model_type'] == 'barlow_twins':
            _up = config.get('use_projector', True)
            if isinstance(_up, list):
                use_projector = bool(_up[0])
            else:
                use_projector = bool(_up)
            if use_projector:
                #   Decoder  ：Decoder(encoder_output_dim, hidden_dims, input_dim)
                projector_out_dim = input_dim
                projector = Projector(encoder_output_dim, hidden_dims[::-1], projector_out_dim).to(device)
                params = list(encoder.parameters()) + list(projector.parameters())
            else:
                projector = None
                params = list(encoder.parameters())
            #   BatchNorm  ：
            # -   projector   projector  （  input_dim）
            # -   projector   encoder_output_dim
            criterion_output_dim = projector_out_dim if use_projector else encoder_output_dim
            criterion = BarlowTwinsLoss(lambda_param, criterion_output_dim).to(device)

        elif config['model_type'] == 'VAE':
            encoder = VAE_Encoder(input_dim, hidden_dims, encoder_output_dim).to(device)
            decoder = Decoder(encoder_output_dim, hidden_dims, input_dim).to(device)
            criterion = nn.MSELoss()
            params = list(encoder.parameters()) + list(decoder.parameters())
        else:
            raise ValueError(f"Unsupported model_type: {config['model_type']}")

        inference_encoder = encoder_online if config['model_type'] == 'byol' else encoder

        optimizer_class = getattr(optim, optimizer)
        optimizer = optimizer_class(
            params,
            lr= learning_rate,
            **config.get('optimizer_params', {})
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['lr_decay_factor'],
            patience=config['lr_patience'],
            min_lr=config['min_lr'],
            verbose=True
        )

        #  （ ）

        #  
        models = {"Encoder": encoder}
        if config['model_type'] == 'auto_encoder' or config['model_type'] == 'VAE':
            models["Decoder"] = decoder
        elif config['model_type'] == 'barlow_twins':
            if projector is not None:
                models["Projector"] = projector
        log_config_and_model(config, models)

        #  
        min_loss = float('inf')
        best_epoch = -1
        losses = []
        embedding_df = None

        logging.info("Stage 3: Training with Gaussian noise distortion...")
        for epoch in range(num_epochs):
            #   train_dataloader（shuffle=True）
            old_lr = optimizer.param_groups[0]['lr']

            if config['model_type'] == 'auto_encoder':
                epoch_loss = train_auto_encoder(encoder, decoder, train_dataloader, criterion, optimizer, scaler,
                                                        device,
                                                        normal_bulk,
                                                        lambda_pseudo_bulk_rate1,
                                                        lambda_pseudo_bulk_rate2,
                                                        regularization,
                                                        l1_regularization_lambda,
                                                        l2_regularization_lambda,
                                                        )
                final_c = None
            elif config['model_type'] == 'VAE':
                epoch_loss = train_vae(encoder, decoder, train_dataloader, criterion, optimizer, scaler,
                                                        device, beta_vae)
                final_c = None
            else:
                #  Gaussian distortion Barlow Twins 
                epoch_loss, last_c = train_barlow_twins(encoder, projector, train_dataloader, criterion, optimizer, scaler,
                                                device,
                                                lambda_pseudo_bulk_rate1,
                                                lambda_pseudo_bulk_rate2,
                                                regularization,
                                                l1_regularization_lambda,
                                                l2_regularization_lambda,
                                                )

                final_c = last_c

            losses.append(epoch_loss)
            scheduler.step(epoch_loss)

            new_lr = optimizer.param_groups[0]['lr']

            #   lr  ，  logger  
            if new_lr != old_lr:
                logging.info(
                    f"LR from {old_lr:.1e} to {new_lr:.1e}"
                )
            print(f"loss: {epoch_loss:.4f}")

            if (epoch + 1) % 100 == 0:
                #  
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}, Learning Rate: {current_lr:.1e}")

            #  
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                best_epoch = epoch + 1
                distortion_embeddings_list = []

                inference_encoder.eval()
                with torch.no_grad():
                    #   DataLoader（shuffle=False） ， 
                    #  ：eval_dataloader   DynamicBarlowDataset(mode='eval')   log2(x+1)
                    #   bulk_batch  ，  encoder。
                    embeddings_list = []
                    for bulk_batch, indices in eval_dataloader:
                        bulk_batch = bulk_batch.to(device)
                        if config['model_type'] == 'VAE':
                             _, embedding_batch, _ = inference_encoder(bulk_batch)
                        else:
                            embedding_batch = inference_encoder(bulk_batch)
                        embeddings_list.append(embedding_batch.cpu())
                    best_embedding = torch.cat(embeddings_list, dim=0)

                    for distortion_tensor in distortions_list:
                        embeddings = []
                        #   batch   embedding
                        for i in range(0, distortion_tensor.shape[0], config['batch_size']):
                            batch = distortion_tensor[i:i + config['batch_size']].to(device)
                            if config['model_type'] == 'VAE':
                                 _, embedding_batch, _ = inference_encoder(batch)
                            else:
                                embedding_batch = inference_encoder(batch)
                            embeddings.append(embedding_batch.cpu())
                        #   distortion_tensor   batch   tensor
                        distortion_embedding = torch.cat(embeddings, dim=0)
                        distortion_embedding_df = pd.DataFrame(distortion_embedding.numpy(), index=eval_bulk.index)
                        distortion_embeddings_list.append(distortion_embedding_df)

                #   bulk_tensor_index  （  eval_dataloader  ）
                embedding_df = pd.DataFrame(best_embedding.numpy(), index=eval_bulk.index)

                #  
                embedding_file = os.path.join(result_dir, "embedding.pt")
                torch.save(embedding_df, embedding_file)

                corr_matrix = embedding_df.corr(method='pearson')  # shape: [embedding_dim, embedding_dim]
                corr_file = os.path.join(result_dir, "embedding_self_correlation.csv")
                corr_matrix.to_csv(corr_file)

                #   numpy array
                C = corr_matrix.values
                dim = C.shape[0]
                diag_deviation = np.abs(np.diag(C) - 1).mean()
                off_diag_mask = ~np.eye(dim, dtype=bool)
                off_diag_deviation = np.abs(C[off_diag_mask]).mean()
                correlation_deviation_score = diag_deviation + off_diag_deviation

                #  
                loss_curve_file = os.path.join(result_dir, "loss_curve.png")
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, epoch + 2), losses, label='Training Loss (Gaussian Noise)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Current Loss Curve with Gaussian Noise Distortion')
                plt.legend()
                plt.grid(True)
                plt.savefig(loss_curve_file)
                plt.close()

                #  
                saved_encoder = inference_encoder if config.get("model_type") == "byol" else encoder
                encoder_file = os.path.join(result_dir, "encoder.pt")
                torch.save(saved_encoder.state_dict(), encoder_file)

                if config['model_type'] == 'auto_encoder' or config['model_type'] == 'VAE':
                    decoder_file = os.path.join(result_dir, "decoder.pt")
                    torch.save(decoder.state_dict(), decoder_file)
                else:
                    if projector is not None:
                        projector_file = os.path.join(result_dir, "projector.pt")
                        torch.save(projector.state_dict(), projector_file)

                logging.info(f"Epoch {best_epoch} with loss: {min_loss:.4f}")

        
        print("Correlation matrix (partial view):")
        print(pd.DataFrame(C).iloc[:5, :5])
        
        #  
        print(f"\nDiagonal deviation: {diag_deviation:.4f}")
        print(f"Off-diagonal deviation: {off_diag_deviation:.4f}")
        print(f"Correlation deviation score: {correlation_deviation_score:.4f}")
        if final_c is not None:
            torch.save(final_c, os.path.join(result_dir, "final_c.pt"))  #  
            logging.info(f"Final correlation matrix shape: {final_c.shape}")
            logging.info(f"Final correlation matrix sample:\n{final_c}")

        #  
        logging.info("Stage 4: Evaluation...")

        result = evaluate_clustering_method(embedding_df,
                                            eval_bulk_labels, eval_survival_df, encoder_output_dim,
                                            distortion_embeddings_list, correlation_deviation_score)

    except Exception as e:
        logging.error(f"Training with Gaussian noise failed: {str(e)}")
        raise


if __name__ == "__main__":
    #  
    use_gaussian_noise = False  #  True Gaussian distortion，False adjacent normal
    
    if use_gaussian_noise:
        print(" Gaussian distortion ...")
        train_and_evaluate_with_gaussian_noise()
    else:
        print(" adjacent normal distortion ...")
        train_and_evaluate()