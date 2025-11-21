"""
Likelihood Module for scARKIDS
===============================
Implements ZINB (Zero-Inflated Negative Binomial) likelihood computation for scRNA-seq data.

Mathematical Background:
-----------------------
ZINB distribution for gene g in cell n of batch b:
    p(x_bng | μ_bng, θ_g, π_bng) = π_bng·δ_0(x_bng) + (1 - π_bng)·NB(x_bng | μ_bng, θ_g)

Where:
    - x_bng: observed count for gene g in cell n of batch b
    - μ_bng: mean expression (from decoder neural network)
    - θ_g: gene-specific dispersion parameter (learnable)
    - π_bng: dropout probability (from decoder neural network)

Parameterization:
    log μ_bng = f_μ^(g)(z_bn^(0), b; θ_μ) + log s_bn
    logit(π_bng) = f_π^(g)(z_bn^(0), b; θ_π)

Decoder likelihood:
    p_θ(x_bn | z_bn^(0), b) = ∏_{g=1}^{G_b} p(x_bng | μ_bng(z_bn^(0), b), θ_g, π_bng(z_bn^(0), b))

Note: All batches share the same gene set (anchor genes), so no masking is required.
"""

from src.utils.logger import Logger
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LikelihoodConfig:
    """
    Configuration for the Likelihood module.
    
    Attributes:
        latent_dim: Dimension of latent space z^(0)
        n_genes: Number of genes (same across all batches)
        n_batches: Number of batches
        hidden_dim: Hidden layer dimension for decoder networks
        n_layers: Number of hidden layers in decoder networks
        dispersion_mode: 'gene' (gene-specific θ_g)
        use_library_size: Whether to include log(s_bn) in mean parameterization
        eps: Small constant for numerical stability
    """
    latent_dim: int
    n_genes: int
    n_batches: int
    hidden_dim: int = 128
    n_layers: int = 2
    dispersion_mode: str = 'gene'
    use_library_size: bool = True
    eps: float = 1e-8
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.n_genes > 0, "n_genes must be positive"
        assert self.n_batches > 0, "n_batches must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.dispersion_mode == 'gene', "Only 'gene' dispersion mode supported"
        assert self.eps > 0, "eps must be positive"


# ============================================================================
# Logger
# ============================================================================




logger = Logger.get_logger(__name__)


# ============================================================================
# Decoder Neural Networks
# ============================================================================

class DecoderMLP(nn.Module):
    """
    Multi-layer perceptron for decoding latent representations.
    
    Architecture:
        Input: [z_bn^(0), batch_onehot] → Hidden layers → Output
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int, 
        n_layers: int
    ):
        """
        Args:
            input_dim: Input dimension (latent_dim + n_batches)
            output_dim: Output dimension (n_genes)
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP
        layers = []
        current_dim = input_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            ])
            current_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)
        
        logger.debug(f"Initialized DecoderMLP: {input_dim} → {hidden_dim}×{n_layers} → {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        h = self.mlp(x)
        out = self.output_layer(h)
        return out


class MeanDecoder(nn.Module):
    """
    Decoder for mean parameter μ_bng.
    
    Parameterization:
        log μ_bng = f_μ(z_bn^(0), b; θ_μ) + log s_bn
    """
    
    def __init__(self, config: LikelihoodConfig):
        super().__init__()
        
        self.config = config
        input_dim = config.latent_dim + config.n_batches
        
        self.decoder = DecoderMLP(
            input_dim=input_dim,
            output_dim=config.n_genes,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers
        )
        
        logger.info(f"Initialized MeanDecoder with input_dim={input_dim}")
    
    def forward(
        self, 
        z: torch.Tensor, 
        batch_onehot: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mean parameter μ_bng.
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            batch_onehot: Batch indicator (batch_size, n_batches)
            library_size: Library size log(s_bn) (batch_size,) or None
        
        Returns:
            Mean μ (batch_size, n_genes), guaranteed positive
        """
        # Concatenate latent and batch
        x = torch.cat([z, batch_onehot], dim=1)
        
        # Decoder output: log-space mean
        log_mu = self.decoder(x)
        
        # Add library size if provided
        if self.config.use_library_size and library_size is not None:
            log_mu = log_mu + library_size.unsqueeze(1)
        
        # Exponentiate to ensure positivity
        mu = torch.exp(log_mu)
        
        return mu


class DropoutDecoder(nn.Module):
    """
    Decoder for dropout probability π_bng.
    
    Parameterization:
        logit(π_bng) = f_π(z_bn^(0), b; θ_π)
    """
    
    def __init__(self, config: LikelihoodConfig):
        super().__init__()
        
        self.config = config
        input_dim = config.latent_dim + config.n_batches
        
        self.decoder = DecoderMLP(
            input_dim=input_dim,
            output_dim=config.n_genes,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers
        )
        
        logger.info(f"Initialized DropoutDecoder with input_dim={input_dim}")
    
    def forward(
        self, 
        z: torch.Tensor, 
        batch_onehot: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dropout probability π_bng.
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            batch_onehot: Batch indicator (batch_size, n_batches)
        
        Returns:
            Dropout probability π in (0, 1) (batch_size, n_genes)
        """
        # Concatenate latent and batch
        x = torch.cat([z, batch_onehot], dim=1)
        
        # Decoder output: logit-space
        logit_pi = self.decoder(x)
        
        # Sigmoid to get probability in (0, 1)
        pi = torch.sigmoid(logit_pi)
        
        return pi


class DispersionParameter(nn.Module):
    """
    Gene-specific dispersion parameter θ_g.
    
    Parameterization:
        θ_g = softplus(θ_raw_g) to ensure positivity
    """
    
    def __init__(self, n_genes: int):
        super().__init__()
        
        # Initialize as ones (softplus inverse of 1 ≈ 0.541)
        self.theta_raw = nn.Parameter(torch.ones(n_genes))
        
        logger.info(f"Initialized DispersionParameter with {n_genes} genes")
    
    def forward(self) -> torch.Tensor:
        """
        Compute positive dispersion parameters.
        
        Returns:
            Dispersion θ (n_genes,), guaranteed positive
        """
        theta = F.softplus(self.theta_raw)
        return theta


# ============================================================================
# ZINB Likelihood
# ============================================================================

def compute_zinb_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute ZINB log-likelihood.
    
    Mathematical formulation:
        p(x | μ, θ, π) = π·δ_0(x) + (1-π)·NB(x | μ, θ)
    
    Where NB is the Negative Binomial:
        NB(x | μ, θ) = Γ(x+θ)/(Γ(θ)Γ(x+1)) · (θ/(θ+μ))^θ · (μ/(θ+μ))^x
    
    Args:
        x: Observed counts (batch_size, n_genes)
        mu: Mean parameter (batch_size, n_genes)
        theta: Dispersion parameter (n_genes,) - broadcasted
        pi: Dropout probability (batch_size, n_genes)
        eps: Small constant for numerical stability
    
    Returns:
        Log-likelihood (batch_size, n_genes)
    """
    try:
        # Ensure numerical stability
        mu = torch.clamp(mu, min=eps)
        theta = torch.clamp(theta, min=eps)
        pi = torch.clamp(pi, min=eps, max=1-eps)
        
        # Negative Binomial log-probability
        # log NB(x | μ, θ) = log Γ(x+θ) - log Γ(θ) - log Γ(x+1)
        #                    + θ·log(θ/(θ+μ)) + x·log(μ/(θ+μ))
        
        t1 = torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1)
        t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
        
        log_nb = t1 + t2 + t3    # This is the recondition
        
        # ZINB log-probability
        # For x = 0: log(π + (1-π)·exp(log_nb_0))
        # For x > 0: log((1-π)·exp(log_nb))
        
        is_zero = (x < eps).float()
        is_nonzero = 1.0 - is_zero
        
        # Case 1: x = 0
        log_pi_case = torch.log(pi + (1 - pi) * torch.exp(log_nb) + eps)
        
        # Case 2: x > 0
        log_nb_case = torch.log(1 - pi + eps) + log_nb
        
        # Combine cases
        log_likelihood = is_zero * log_pi_case + is_nonzero * log_nb_case    # This is the loss
        
        return log_likelihood
        
    except Exception as e:
        logger.error(f"Error in ZINB likelihood computation: {e}")
        raise


# ============================================================================
# Likelihood Module
# ============================================================================

class LikelihoodModule(nn.Module):
    """
    Complete likelihood module implementing ZINB distribution.
    
    Computes:
        p_θ(x_bn | z_bn^(0), b) = ∏_{g=1}^G p(x_bng | μ_bng, θ_g, π_bng)
    """
    
    def __init__(self, config: LikelihoodConfig):
        """
        Initialize likelihood module.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        
        # Decoder networks
        self.mean_decoder = MeanDecoder(config)
        self.dropout_decoder = DropoutDecoder(config)
        
        # Dispersion parameters
        self.dispersion = DispersionParameter(config.n_genes)
        
        logger.info("Initialized LikelihoodModule")
        logger.info(f"  latent_dim={config.latent_dim}")
        logger.info(f"  n_genes={config.n_genes}")
        logger.info(f"  n_batches={config.n_batches}")
        logger.info(f"  hidden_dim={config.hidden_dim}")
        logger.info(f"  n_layers={config.n_layers}")
    
    def forward(
        self,
        z: torch.Tensor,
        batch_onehot: torch.Tensor,
        x: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute negative log-likelihood (loss).
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            batch_onehot: Batch indicator (batch_size, n_batches)
            x: Observed counts (batch_size, n_genes)
            library_size: Library size log(s_bn) (batch_size,) or None
        
        Returns:
            loss: Negative log-likelihood (scalar)
            outputs: Dictionary containing:
                - mu: Mean parameter (batch_size, n_genes)
                - pi: Dropout probability (batch_size, n_genes)
                - theta: Dispersion parameter (n_genes,)
                - log_likelihood: Per-gene log-likelihood (batch_size, n_genes)
        """
        try:
            # Decode parameters
            mu = self.mean_decoder(z, batch_onehot, library_size)
            pi = self.dropout_decoder(z, batch_onehot)
            theta = self.dispersion()
            
            # Compute log-likelihood
            log_likelihood = compute_zinb_log_likelihood(
                x, mu, theta, pi, eps=self.config.eps
            )
            
            # Negative log-likelihood loss (summed over genes, averaged over batch)
            loss = -log_likelihood.sum(dim=1).mean()
            
            # Prepare outputs
            outputs = {
                'mu': mu,
                'pi': pi,
                'theta': theta,
                'log_likelihood': log_likelihood
            }
            
            return loss, outputs
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise


# ============================================================================
# Manager Class (Module Entry Point)
# ============================================================================

class LikelihoodManager:
    """
    Manager class for the Likelihood module.
    
    This is the single entry point that:
    1. Parses configuration from config.yaml
    2. Initializes the likelihood module
    3. Exposes APIs for training/inference
    """
    
    def __init__(self, config_dict: Dict):
        """
        Initialize manager from configuration dictionary.
        
        Args:
            config_dict: Dictionary containing 'likelihood' section from config.yaml
        """
        logger.info("Initializing LikelihoodManager")
        
        # Parse configuration
        self.config = self._parse_config(config_dict)
        
        # Initialize likelihood module
        self.likelihood_module = LikelihoodModule(self.config)
        
        logger.info("LikelihoodManager initialized successfully")
    
    def _parse_config(self, config_dict: Dict) -> LikelihoodConfig:
        """
        Parse configuration dictionary into LikelihoodConfig.
        
        Args:
            config_dict: Dictionary from config.yaml
        
        Returns:
            LikelihoodConfig object
        """
        try:
            config = LikelihoodConfig(
                latent_dim=config_dict['latent_dim'],
                n_genes=config_dict['n_genes'],
                n_batches=config_dict['n_batches'],
                hidden_dim=config_dict.get('hidden_dim', 128),
                n_layers=config_dict.get('n_layers', 2),
                dispersion_mode=config_dict.get('dispersion_mode', 'gene'),
                use_library_size=config_dict.get('use_library_size', True),
                eps=config_dict.get('eps', 1e-8)
            )
            
            logger.info("Configuration parsed successfully")
            return config
            
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
    
    def get_module(self) -> LikelihoodModule:
        """
        Get the likelihood module. This is the main class of the module.
        
        Returns:
            LikelihoodModule instance
        """
        return self.likelihood_module
    
    def compute_likelihood(
        self,
        z: torch.Tensor,
        batch_onehot: torch.Tensor,
        x: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute likelihood (main API for training/inference).
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            batch_onehot: Batch indicator (batch_size, n_batches)
            x: Observed counts (batch_size, n_genes)
            library_size: Library size log(s_bn) (batch_size,) or None
        
        Returns:
            loss: Negative log-likelihood (scalar)
            outputs: Dictionary containing decoded parameters and log-likelihood
        """
        return self.likelihood_module(z, batch_onehot, x, library_size)
    
    def get_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get all trainable parameters.
        
        Returns:
            Dictionary of named parameters
        """
        return dict(self.likelihood_module.named_parameters())


# ============================================================================
# Config YAML Schema Documentation
# ============================================================================

"""
Example usage:
--------------
```python
import yaml
from likelihood import LikelihoodManager

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize manager
manager = LikelihoodManager(config['likelihood'])

# Get module for training
likelihood_module = manager.get_module()

# Compute loss
loss, outputs = manager.compute_likelihood(z, batch_onehot, x, library_size)
```
"""
