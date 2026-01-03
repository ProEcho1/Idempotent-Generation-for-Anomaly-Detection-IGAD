<h1 align="center">Multivariate Time Series Anomaly Detection with Idempotent Reconstruction [NeurIPS 2025]</h1>

<p align="center"> 
  <a><img src="https://img.shields.io/github/license/ProEcho1/Idempotent-Generation-for-Anomaly-Detection-IGAD"></a>
  <a><img src="https://img.shields.io/github/last-commit/ProEcho1/Idempotent-Generation-for-Anomaly-Detection-IGAD"></a>
  <a><img src="https://img.shields.io/badge/Python-3.8.13-red"></a>
</p> 

## 🚀 **The official repository for the paper** ***Multivariate Time Series Anomaly Detection with Idempotent Reconstruction***

Reconstruction-based methods are competitive choices for multivariate time series anomaly detection (MTS AD). However, one challenge these methods may suffer is over generalization, where abnormal inputs are also well reconstructed. In addition, balancing robustness and sensitivity is also important for final performance, as robustness ensures accurate detection in potentially noisy data, while sensitivity enables early detection of subtle anomalies. To address these problems, inspired by idempotent generative network, we take the view from the manifold and propose a novel module named **I**dempotent **G**eneration for **A**nomaly**D**etection (IGAD) which can be flexibly combined with a reconstruction-based method without introducing additional trainable parameters. We modify the manifold to make sure that normal time points can be mapped onto it while tightening it to drop out abnormal time points simultaneously. Regarding the latest findings of AD metrics, we evaluated IGAD on various methods with four real-world datasets, and they achieve visible improvements in VUS-PR than their predecessors, demonstrating the effective potential of IGAD for further improvements in MTS AD tasks.

![The architecture of IGAD](./figures/Architecture.jpg)

# 🚩 IGAD Integration Guide

This guide provides a comprehensive walkthrough for integrating the IGAD module into a custom multivariate time series anomaly detection model.

It abstracts the specific implementation details (like those in FITS or DAGMM) into a general pattern: separating the Neural Network definition (e.g., `nn.Module`) from the Solver/Trainer class (where the training loop resides).

## 1. The IGAD Module

Save the following code as `igad_module.py` in your project's utility folder (e.g., `models/utils/`). This class encapsulates the logic for frequency domain augmentation, idempotent consistency, and manifold tightening.

> **Note:** This implementation corresponds to the general form found in `IGN_Ordinary.py`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IdempotentLoss(nn.Module):
    def __init__(self, model, idem_weight, tight_weight, loss_tight_clamp_ratio):
        """
        Initialize the IGAD Loss Module.
        
        Args:
            model (nn.Module): A frozen copy of the training model structure.
            idem_weight (float): Weight for the Idempotent Loss (lambda_idem).
            tight_weight (float): Weight for the Tightness Loss (lambda_tight).
            loss_tight_clamp_ratio (float): Smooth clamping ratio for Tightness Loss (alpha).
        """
        super().__init__()
        self.training_model_copy = model
        self.idem_weight = idem_weight
        self.tight_weight = tight_weight
        self.loss_tight_clamp_ratio = loss_tight_clamp_ratio
        
        # Ensure the copy is frozen to save resources
        for param in self.training_model_copy.parameters():
            param.requires_grad = False

    def forward(self, input_data, output_data, training_model):
        """
        Calculate the IGAD combined loss.

        Args:
            input_data (Tensor): Original input batch (Batch, Length, Feature).
            output_data (Tensor): Reconstructed output (Batch, Length, Feature).
            training_model (nn.Module): The active model currently being updated.
            
        Returns:
            Tensor: The weighted IGAD loss (Scalar).
        """
        cur_batch_size = input_data.shape[0]

        # 1. Calculate basic reconstruction loss for thresholding (not for optimization)
        loss_rec = F.mse_loss(input_data, output_data, reduction='none').reshape(cur_batch_size, -1).mean(dim=-1)

        # 2. Synchronize the frozen copy with the current model state
        self.training_model_copy.load_state_dict(training_model.state_dict())

        # 3. Prepare data for FFT (Batch, Length, Feature) -> (Batch, Feature, Length)
        idem_data = input_data.permute(0, 2, 1)

        # 4. Generate augmented pseudo-normal samples (z)
        freq_means_and_stds = torch.stack(self.get_freq_means_and_stds(idem_data)).unsqueeze(0)
        num_dims = len(freq_means_and_stds.shape) - 1
        freq_means_and_stds = freq_means_and_stds.repeat(idem_data.shape[0], *(1,) * num_dims).unbind(dim=1)
        
        z = self.get_noise(*freq_means_and_stds)
        z = z.permute(0, 2, 1)  # Restore shape to (Batch, Length, Feature)

        # 5. Idempotent Manifold Mapping
        fz = training_model(z)                  # f(z)
        f_z = fz.detach()                       # Detach for target
        
        ff_z = training_model(f_z)              # f(f(z).detach()) -> approximates f(f'(z)) behavior
        f_fz = self.training_model_copy(fz)     # f'(f(z))

        # 6. Compute Losses
        # Idempotent Loss: Enforce f'(f(z)) == f(z)
        loss_idem = F.l1_loss(f_fz, fz, reduction='mean')

        # Tightness Loss: Push f(f'(z)) away from f'(z)
        loss_tight = -F.l1_loss(ff_z, f_z, reduction='none').reshape(cur_batch_size, -1).mean(dim=-1)
        
        # Smooth Clamping using tanh
        loss_tight_clamp = self.loss_tight_clamp_ratio * loss_rec
        loss_tight_clamp = torch.clamp(loss_tight_clamp, min=1e-6) # Avoid division by zero
        loss_tight = torch.tanh(loss_tight / loss_tight_clamp) * loss_tight_clamp
        
        loss_rec = loss_rec.mean()
        loss_tight = loss_tight.mean()

        # 7. Total Weighted Loss
        loss = self.idem_weight * loss_idem + self.tight_weight * loss_tight
        return loss

    def get_freq_means_and_stds(self, x):
        freq = torch.fft.fft(x, dim=-1)
        real_mean = freq.real.mean(dim=0)
        real_std = freq.real.std(dim=0)
        imag_mean = freq.imag.mean(dim=0)
        imag_std = freq.imag.std(dim=0)
        return real_mean, real_std, imag_mean, imag_std

    def get_noise(self, real_mean, real_std, imag_mean, imag_std):
        freq_real = torch.normal(real_mean, real_std)
        freq_imag = torch.normal(imag_mean, imag_std)
        freq = freq_real + 1j * freq_imag
        noise = torch.fft.ifft(freq, dim=-1)
        return noise.real
```

## 2. Integration Guide

This section demonstrates how to modify your Solver/Trainer Class (e.g., the DAGMM or FITS class in your files) to use IGAD.

### Phase 1: Preparation (The `__init__` Method)

In your solver class initialization, you need to instantiate a frozen copy of your model and the IGAD Loss module.

```python
import copy
from igad_module import IdempotentLoss

class AnomalyDetector:
    def __init__(self, config, ...):
        # -----------------------------------------------------------
        # 1. Existing Model Initialization
        # -----------------------------------------------------------
        self.device = self._get_device()
        # YourBaseModel is your nn.Module (e.g., DAGMMModel, FITS Model)
        self.model = YourBaseModel(config).to(self.device)

        # -----------------------------------------------------------
        # 2. IGAD Integration: Create Frozen Copy
        # -----------------------------------------------------------
        # Initialize an identical model structure
        self.model_copy = YourBaseModel(config).to(self.device)
        
        # Explicitly freeze it (best practice from FITS_IGAD.py)
        self.model_copy.requires_grad_(False)

        # -----------------------------------------------------------
        # 3. IGAD Integration: Initialize Loss Module
        # -----------------------------------------------------------
        # Hyperparameters (Recommended defaults)
        self.idem_weight = 0.1  # lambda_idem
        self.tight_weight = 0.1 # lambda_tight
        self.alpha = 1.1        # loss_tight_clamp_ratio

        self.ign = IdempotentLoss(
            model=self.model_copy,
            idem_weight=self.idem_weight,
            tight_weight=self.tight_weight,
            loss_tight_clamp_ratio=self.alpha
        )

        # -----------------------------------------------------------
        # 4. Optimizer Setup (Only optimize the main model)
        # -----------------------------------------------------------
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.rec_criterion = torch.nn.MSELoss()
```

### Phase 2: Training Logic (The `fit` Method)

You must modify the training loop to include the IGAD loss calculation.

> **Critical Implementation Detail:** Always `clone()` the input data before passing it to IGAD. This ensures that if your model modifies the input in-place (or performs downsampling/masking), IGAD still operates on the correct data distribution.

```python
    def fit(self, train_loader):
        self.model.train()
        
        for epoch in range(self.epochs):
            loop = tqdm.tqdm(train_loader)
            
            for i, (input_data, target_data) in enumerate(loop):
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                # -------------------------------------------------------
                # 1. Create Data Backup for IGAD
                # -------------------------------------------------------
                input_backup = input_data.clone()

                self.optimizer.zero_grad()

                # -------------------------------------------------------
                # 2. Forward Pass (Main Task)
                # -------------------------------------------------------
                # Depending on your model, this might return multiple values
                # (e.g., reconstruction, latent vars, etc.)
                output, _ = self.model(input_data)

                # -------------------------------------------------------
                # 3. Calculate IGAD Loss
                # -------------------------------------------------------
                # Pass the BACKUP data, the OUTPUT, and the MODEL instance
                igad_loss_value = self.ign(
                    input_data=input_backup, 
                    output_data=output, 
                    training_model=self.model
                )

                # -------------------------------------------------------
                # 4. Calculate Main Reconstruction Loss
                # -------------------------------------------------------
                main_loss = self.rec_criterion(output, target_data)

                # -------------------------------------------------------
                # 5. Total Loss & Optimization
                # -------------------------------------------------------
                total_loss = main_loss + igad_loss_value
                
                total_loss.backward()
                self.optimizer.step()

                # Logging
                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(
                    total=total_loss.item(), 
                    rec=main_loss.item(), 
                    igad=igad_loss_value.item()
                )
```

### Phase 3: Validation (Optional but Recommended)

Including IGAD loss in validation helps you track if the manifold constraints are being learned correctly without overfitting.

```python
    def validate(self, valid_loader):
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for input_data, target_data in valid_loader:
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                input_backup = input_data.clone()
                
                output, _ = self.model(input_data)
                
                # Calculate IGAD loss in eval mode to track convergence
                igad_loss_value = self.ign(
                    input_data=input_backup, 
                    output_data=output, 
                    training_model=self.model
                )
                
                main_loss = self.rec_criterion(output, target_data)
                total_loss = main_loss + igad_loss_value
                total_val_loss += total_loss.item()
                
        return total_val_loss / len(valid_loader)
```

# 🚩 Run with Examples (Uploading....)

```bash
  bash train.sh
```
