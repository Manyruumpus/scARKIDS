"""
Prior Module for scARKIDS
==========================

Implements prior distributions for VAE-DDPM models (supervised and unsupervised).

Mathematical Background:
-----------------------

Supervised Mode:
  - VAE Latent: p(z^(0)|c*) = N(z^(0)|μ_c, Σ_c) with learnable μ_c, Σ_c
  - Cell Type: p(c) = Categorical(c|π_0)
  - Terminal Diffusion: p(z^(T)) = N(z^(T)|0, I)
  - Batch: p(b) = Categorical(b|ρ)

Unsupervised Mode:
  - VAE Latent: p(z^(0)) = N(z^(0)|0, I) - shared standard Gaussian
  - Cell Type: p(c) = Categorical(c|π_0)
  - Terminal Diffusion: p(z^(T)) = N(z^(T)|0, I)
  - Batch: p(b) = Categorical(b|ρ)

Key Difference:
  Supervised uses cell-type-specific learned Gaussians; unsupervised uses fixed N(0,I).

Note: All batches share the same gene set (anchor genes).
"""

from src.utils.logger import Logger
from typing import Dict, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PriorConfig:
    """
    Configuration for the Prior module.
    
    Attributes:
        latent_dim: Dimension of latent space z^(0)
        n_cell_types: Number of cell types
        n_batches: Number of batches
        supervised: Whether to use supervised (True) or unsupervised (False) mode
        cell_type_prior_probs: Optional prior probabilities for cell types (default: uniform)
        batch_prior_probs: Optional prior probabilities for batches (default: uniform)
    """
    latent_dim: int
    n_cell_types: int
    n_batches: int
    supervised: bool
    cell_type_prior_probs: Optional[list] = None
    batch_prior_probs: Optional[list] = None
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.n_cell_types > 0, "n_cell_types must be positive"
        assert self.n_batches > 0, "n_batches must be positive"
        
        # Validate cell type prior if provided
        if self.cell_type_prior_probs is not None:
            assert len(self.cell_type_prior_probs) == self.n_cell_types, \
                f"cell_type_prior_probs length mismatch: {len(self.cell_type_prior_probs)} vs {self.n_cell_types}"
            assert abs(sum(self.cell_type_prior_probs) - 1.0) < 1e-6, \
                "cell_type_prior_probs must sum to 1"
        
        # Validate batch prior if provided
        if self.batch_prior_probs is not None:
            assert len(self.batch_prior_probs) == self.n_batches, \
                f"batch_prior_probs length mismatch: {len(self.batch_prior_probs)} vs {self.n_batches}"
            assert abs(sum(self.batch_prior_probs) - 1.0) < 1e-6, \
                "batch_prior_probs must sum to 1"

# ============================================================================
# Logger
# ============================================================================

logger = Logger.get_logger(__name__)

# ============================================================================
# Prior Distributions
# ============================================================================

class VAELatentPriorSupervised(nn.Module):
    """
    Cell-type-specific VAE latent prior: p(z^(0)|c*) = N(z^(0)|μ_c, Σ_c)
    
    Learnable parameters μ_c and Σ_c for each cell type.
    Different cell types occupy different regions of latent space.
    """
    def __init__(self, latent_dim: int, n_cell_types: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_cell_types = n_cell_types
        
        # Learnable cell-type-specific means (n_cell_types, latent_dim)
        self.means = nn.Parameter(torch.randn(n_cell_types, latent_dim))
        
        # Learnable cell-type-specific log-variances (n_cell_types, latent_dim)
        # Using log-variance for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(n_cell_types, latent_dim))
        
        logger.info(f"Initialized VAELatentPriorSupervised: {n_cell_types} cell types, latent_dim={latent_dim}")
    
    def log_prob(self, z: torch.Tensor, cell_type: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z^(0)|c*) for cell-type-specific Gaussian.
        
        Args:
            z: Latent codes (batch_size, latent_dim)
            cell_type: Cell type indices (batch_size,), dtype=long
        
        Returns:
            Log probabilities (batch_size,)
        """
        # Get cell-type-specific parameters
        means = self.means[cell_type]  # (batch_size, latent_dim)
        log_vars = self.log_vars[cell_type]  # (batch_size, latent_dim)
        vars = torch.exp(log_vars)
        
        # Gaussian log probability: log N(z|μ,σ²) = -0.5 * [Σ(z-μ)²/σ² + Σlog(σ²) + d*log(2π)]
        diff = z - means
        log_prob = -0.5 * (
            torch.sum(diff ** 2 / vars, dim=1) +
            torch.sum(log_vars, dim=1) +
            self.latent_dim * np.log(2 * np.pi)
        )
        return log_prob
    
    def sample(self, n_samples: int, cell_type: torch.Tensor) -> torch.Tensor:
        """
        Sample from p(z^(0)|c*) using reparameterization trick.
        
        Args:
            n_samples: Number of samples (should match cell_type.shape[0])
            cell_type: Cell type indices (n_samples,), dtype=long
        
        Returns:
            Samples (n_samples, latent_dim)
        """
        means = self.means[cell_type]
        stds = torch.exp(0.5 * self.log_vars[cell_type])
        epsilon = torch.randn_like(means)
        return means + stds * epsilon


class VAELatentPriorUnsupervised(nn.Module):
    """
    Shared VAE latent prior: p(z^(0)) = N(z^(0)|0, I)
    
    Standard Gaussian (no learnable parameters).
    Cell types are inferred through a separate classifier.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        logger.info(f"Initialized VAELatentPriorUnsupervised: standard Gaussian, latent_dim={latent_dim}")
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z^(0)) = log N(z^(0)|0, I).
        
        Args:
            z: Latent codes (batch_size, latent_dim)
        
        Returns:
            Log probabilities (batch_size,)
        """
        # log N(z|0,I) = -0.5 * [||z||² + d*log(2π)]
        log_prob = -0.5 * (
            torch.sum(z ** 2, dim=1) +
            self.latent_dim * np.log(2 * np.pi)
        )
        return log_prob
    
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from N(0, I).
        
        Args:
            n_samples: Number of samples
            device: Device to create tensor on
        
        Returns:
            Samples (n_samples, latent_dim)
        """
        return torch.randn(n_samples, self.latent_dim, device=device)


class CellTypePrior(nn.Module):
    """
    Cell type prior: p(c) = Categorical(c|π_0)
    
    Typically uniform π_0 = (1/C, ..., 1/C).
    Can be initialized from empirical frequencies.
    """
    def __init__(self, n_cell_types: int, probabilities: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_cell_types = n_cell_types
        
        # Initialize probabilities
        if probabilities is None:
            probabilities = torch.ones(n_cell_types) / n_cell_types
            logger.info(f"CellTypePrior: uniform distribution over {n_cell_types} cell types")
        else:
            logger.info(f"CellTypePrior: empirical frequencies over {n_cell_types} cell types")
        
        # Register as buffer (not learnable)
        self.register_buffer("probabilities", probabilities)
    
    def log_prob(self, c: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(c).
        
        Args:
            c: Cell type indices (batch_size,), dtype=long
        
        Returns:
            Log probabilities (batch_size,)
        """
        return torch.log(self.probabilities[c])
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from Categorical(π_0).
        
        Args:
            n_samples: Number of samples
        
        Returns:
            Cell type indices (n_samples,)
        """
        samples = torch.multinomial(
            self.probabilities.unsqueeze(0).expand(n_samples, -1),
            num_samples=1
        ).squeeze(1)
        return samples


class TerminalDiffusionPrior(nn.Module):
    """
    Terminal diffusion prior: p(z^(T)) = N(z^(T)|0, I)
    
    Standard Gaussian at final diffusion step (no learnable parameters).
    Same for both supervised and unsupervised modes.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        logger.info(f"Initialized TerminalDiffusionPrior: standard Gaussian, latent_dim={latent_dim}")
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z^(T)) = log N(z^(T)|0, I).
        
        Args:
            z: Terminal latent codes (batch_size, latent_dim)
        
        Returns:
            Log probabilities (batch_size,)
        """
        log_prob = -0.5 * (
            torch.sum(z ** 2, dim=1) +
            self.latent_dim * np.log(2 * np.pi)
        )
        return log_prob
    
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from N(0, I).
        
        Args:
            n_samples: Number of samples
            device: Device to create tensor on
        
        Returns:
            Samples (n_samples, latent_dim)
        """
        return torch.randn(n_samples, self.latent_dim, device=device)


class BatchPrior(nn.Module):
    """
    Batch prior: p(b) = Categorical(b|ρ)
    
    Typically uniform or empirical frequencies.
    """
    def __init__(self, n_batches: int, probabilities: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_batches = n_batches
        
        # Initialize probabilities
        if probabilities is None:
            probabilities = torch.ones(n_batches) / n_batches
            logger.info(f"BatchPrior: uniform distribution over {n_batches} batches")
        else:
            logger.info(f"BatchPrior: empirical frequencies over {n_batches} batches")
        
        # Register as buffer (not learnable)
        self.register_buffer("probabilities", probabilities)
    
    def log_prob(self, b: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(b).
        
        Args:
            b: Batch indices (batch_size,), dtype=long
        
        Returns:
            Log probabilities (batch_size,)
        """
        return torch.log(self.probabilities[b])
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from Categorical(ρ).
        
        Args:
            n_samples: Number of samples
        
        Returns:
            Batch indices (n_samples,)
        """
        samples = torch.multinomial(
            self.probabilities.unsqueeze(0).expand(n_samples, -1),
            num_samples=1
        ).squeeze(1)
        return samples


# ============================================================================
# Prior Module
# ============================================================================

class PriorModule(nn.Module):
    """
    Complete prior module for VAE-DDPM.
    
    Composes:
      - VAE latent prior (supervised or unsupervised)
      - Cell type prior
      - Terminal diffusion prior
      - Batch prior
    """
    def __init__(self, config: PriorConfig):
        """
        Initialize prior module.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # VAE latent prior (supervised or unsupervised)
        if config.supervised:
            self.vae_latent_prior = VAELatentPriorSupervised(
                config.latent_dim, config.n_cell_types
            )
        else:
            self.vae_latent_prior = VAELatentPriorUnsupervised(config.latent_dim)
        
        # Cell type prior
        cell_type_probs = None
        if config.cell_type_prior_probs is not None:
            cell_type_probs = torch.tensor(config.cell_type_prior_probs, dtype=torch.float32)
        self.cell_type_prior = CellTypePrior(config.n_cell_types, cell_type_probs)
        
        # Terminal diffusion prior
        self.terminal_diffusion_prior = TerminalDiffusionPrior(config.latent_dim)
        
        # Batch prior
        batch_probs = None
        if config.batch_prior_probs is not None:
            batch_probs = torch.tensor(config.batch_prior_probs, dtype=torch.float32)
        self.batch_prior = BatchPrior(config.n_batches, batch_probs)
        
        logger.info("Initialized PriorModule")
        logger.info(f"  mode={'supervised' if config.supervised else 'unsupervised'}")
        logger.info(f"  latent_dim={config.latent_dim}")
        logger.info(f"  n_cell_types={config.n_cell_types}")
        logger.info(f"  n_batches={config.n_batches}")
    
    def log_prob_vae_latent(
        self,
        z: torch.Tensor,
        cell_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log p(z^(0)|c*) or log p(z^(0)) depending on mode.
        
        Args:
            z: Latent codes (batch_size, latent_dim)
            cell_type: Cell type indices (batch_size,), required for supervised mode
        
        Returns:
            Log probabilities (batch_size,)
        """
        if self.config.supervised:
            assert cell_type is not None, "cell_type required for supervised mode"
            return self.vae_latent_prior.log_prob(z, cell_type)
        else:
            return self.vae_latent_prior.log_prob(z)
    
    def log_prob_cell_type(self, c: torch.Tensor) -> torch.Tensor:
        """Compute log p(c)"""
        return self.cell_type_prior.log_prob(c)
    
    def log_prob_terminal_diffusion(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z^(T))"""
        return self.terminal_diffusion_prior.log_prob(z)
    
    def log_prob_batch(self, b: torch.Tensor) -> torch.Tensor:
        """Compute log p(b)"""
        return self.batch_prior.log_prob(b)
    
    def sample_joint_prior(
        self,
        n_samples: int,
        device: torch.device,
        cell_type: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from joint prior distribution.
        
        Args:
            n_samples: Number of samples
            device: Device to create tensors on
            cell_type: Optional cell type indices (n_samples,)
            batch_idx: Optional batch indices (n_samples,)
        
        Returns:
            Dictionary containing:
                - z_0: VAE latent samples (n_samples, latent_dim)
                - c: Cell type samples (n_samples,)
                - b: Batch samples (n_samples,)
                - z_T: Terminal diffusion samples (n_samples, latent_dim)
        """
        # Sample cell types if not provided
        if cell_type is None:
            cell_type = self.cell_type_prior.sample(n_samples).to(device)
        
        # Sample batches if not provided
        if batch_idx is None:
            batch_idx = self.batch_prior.sample(n_samples).to(device)
        
        # Sample VAE latent
        if self.config.supervised:
            z_0 = self.vae_latent_prior.sample(n_samples, cell_type)
        else:
            z_0 = self.vae_latent_prior.sample(n_samples, device)
        
        # Sample terminal diffusion
        z_T = self.terminal_diffusion_prior.sample(n_samples, device)
        
        return {
            "z_0": z_0,
            "c": cell_type,
            "b": batch_idx,
            "z_T": z_T
        }


# ============================================================================
# Manager Class (Module Entry Point)
# ============================================================================

class PriorManager:
    """
    Manager class for the Prior module.
    
    This is the single entry point that:
    1. Parses configuration from config.yaml
    2. Initializes the prior module
    3. Exposes APIs for training/inference
    """
    def __init__(self, config_dict: Dict):
        """
        Initialize manager from configuration dictionary.
        
        Args:
            config_dict: Dictionary containing 'prior' section from config.yaml
        """
        logger.info("Initializing PriorManager")
        
        # Parse configuration
        self.config = self._parse_config(config_dict)
        
        # Initialize prior module
        self.prior_module = PriorModule(self.config)
        
        logger.info("PriorManager initialized successfully")
    
    def _parse_config(self, config_dict: Dict) -> PriorConfig:
        """
        Parse configuration dictionary into PriorConfig.
        
        Args:
            config_dict: Dictionary from config.yaml
        
        Returns:
            PriorConfig object
        """
        try:
            config = PriorConfig(
                latent_dim=config_dict['latent_dim'],
                n_cell_types=config_dict['n_cell_types'],
                n_batches=config_dict['n_batches'],
                supervised=config_dict['supervised'],
                cell_type_prior_probs=config_dict.get('cell_type_prior_probs', None),
                batch_prior_probs=config_dict.get('batch_prior_probs', None)
            )
            logger.info("Configuration parsed successfully")
            return config
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
    
    def get_module(self) -> PriorModule:
        """
        Get the prior module.
        
        Returns:
            PriorModule instance
        """
        return self.prior_module
    
    def log_prob_vae_latent(
        self,
        z: torch.Tensor,
        cell_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log p(z^(0)|c*) or log p(z^(0))"""
        return self.prior_module.log_prob_vae_latent(z, cell_type)
    
    def log_prob_cell_type(self, c: torch.Tensor) -> torch.Tensor:
        """Compute log p(c)"""
        return self.prior_module.log_prob_cell_type(c)
    
    def log_prob_terminal_diffusion(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z^(T))"""
        return self.prior_module.log_prob_terminal_diffusion(z)
    
    def log_prob_batch(self, b: torch.Tensor) -> torch.Tensor:
        """Compute log p(b)"""
        return self.prior_module.log_prob_batch(b)
    
    def sample_joint_prior(
        self,
        n_samples: int,
        device: torch.device,
        cell_type: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Sample from joint prior"""
        return self.prior_module.sample_joint_prior(n_samples, device, cell_type, batch_idx)
    
    def get_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get all trainable parameters.
        
        Returns:
            Dictionary of named parameters
        """
        return dict(self.prior_module.named_parameters())


# ============================================================================
# Config YAML Schema Documentation
# ============================================================================

"""
Example usage:
--------------

```python
import yaml
from prior import PriorManager

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize manager
manager = PriorManager(config['prior'])

# Get module for training
prior_module = manager.get_module()

# Compute log probabilities
log_p_z0 = manager.log_prob_vae_latent(z, cell_type=c)  # supervised
log_p_c = manager.log_prob_cell_type(c)
log_p_zT = manager.log_prob_terminal_diffusion(z_T)

# Sample from prior
samples = manager.sample_joint_prior(n_samples=100, device=torch.device('cuda'))
```
"""
