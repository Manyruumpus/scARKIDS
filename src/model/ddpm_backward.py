"""
scARKIDS DDPM Backward (Reverse) Process Module

Implements the reverse (denoising) diffusion process for both supervised and
unsupervised VAE-DDPM models.

Mathematical Framework:

=== SUPERVISED MODE ===
Reverse process conditioned on known cell type c*:

Full trajectory:
    p_ψ(z^(0:T) | c*) = p(z^(T)) ∏_{t=1}^{T} p_ψ(z^(t-1) | z^(t), c*)

Single reverse step:
    p_ψ(z^(t-1) | z^(t), c*) = N(z^(t-1) | μ_ψ(z^(t), t, c*), Σ_ψ(z^(t), t, c*))

Parameterization (noise prediction formulation):
    μ_ψ(z^(t), t, c*) = (1/√α_t) * (z^(t) - (β_t/√(1-ᾱ_t)) * ε_ψ(z^(t), t, c*))

where ε_ψ is a neural network predicting noise.

Variance: Σ_ψ = σ_t² I
where σ_t² = β_t or learned.

=== UNSUPERVISED MODE ===
Reverse process conditioned on predicted (random) cell type c:

Full trajectory:
    p_ψ(z^(0:T)) = p(z^(T)) ∏_{t=1}^{T} p_ψ(z^(t-1) | z^(t), c)

Key difference: c is now a random variable (not fixed).

Single reverse step:
    p_ψ(z^(t-1) | z^(t), c) = N(z^(t-1) | μ_ψ(z^(t), t, c), Σ_ψ(z^(t), t, c))

Parameterization (same form as supervised):
    μ_ψ(z^(t), t, c) = (1/√α_t) * (z^(t) - (β_t/√(1-ᾱ_t)) * ε_ψ(z^(t), t, c))

=== KEY COMPONENTS ===

1. Noise Prediction Network ε_ψ(z^(t), t, c):
   - Input: noisy latent z^(t), timestep t, cell type c
   - Output: predicted noise ε̂
   - Architecture: MLP with timestep and cell type embeddings

2. Mean Prediction:
   - Formula: μ_ψ = (1/√α_t) * (z^(t) - (β_t/√(1-ᾱ_t)) * ε̂)
   - Computed using variance schedule from forward process

3. Variance Prediction:
   - Fixed: σ_t² = β_t (standard DDPM)
   - Learnable: σ_t² can be learned (improved DDPM)
   - Variance Σ_ψ = σ_t² I (isotropic Gaussian)

4. Sampling Process:
   - Start from z^(T) ~ N(0, I)
   - For t = T, T-1, ..., 1:
       - Predict noise: ε̂ = ε_ψ(z^(t), t, c)
       - Compute mean: μ = (1/√α_t) * (z^(t) - (β_t/√(1-ᾱ_t)) * ε̂)
       - Sample: z^(t-1) = μ + σ_t * ε', where ε' ~ N(0, I)
   - Return z^(0)

=== DESIGN PRINCIPLES ===
1. No union gene set masking (all batches have same genes)
2. Unified architecture for supervised and unsupervised modes
3. Noise prediction formulation for better training stability
4. Support for both fixed and learned variance
5. OOP design with clear separation of concerns
6. Comprehensive logging and error handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

from src.utils.logger import Logger

logger = Logger.get_logger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DDPMBackwardConfig:
    """Configuration for DDPM backward process with validation."""
    
    latent_dim: int                    # Dimension of latent space z
    n_diffusion_steps: int             # Number of diffusion steps T
    n_cell_types: int                  # Number of cell types C
    
    # Noise network architecture
    noise_network_hidden_dim: int = 256     # Hidden layer dimension
    noise_network_n_layers: int = 3         # Number of hidden layers
    noise_network_dropout: float = 0.1      # Dropout probability
    
    # Timestep embedding
    timestep_embed_dim: int = 128      # Dimension of timestep embedding
    
    # Cell type embedding
    celltype_embed_dim: int = 64       # Dimension of cell type embedding
    
    # Variance strategy
    variance_type: str = "fixed"       # "fixed" or "learned"
    
    # Other settings
    device: str = "cpu"
    
    def __post_init__(self):
        """Validate all parameters."""
        
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")
        
        if self.n_diffusion_steps <= 0:
            raise ValueError(
                f"n_diffusion_steps must be > 0, got {self.n_diffusion_steps}"
            )
        
        if self.n_cell_types <= 0:
            raise ValueError(
                f"n_cell_types must be > 0, got {self.n_cell_types}"
            )
        
        if self.noise_network_hidden_dim <= 0:
            raise ValueError(
                f"noise_network_hidden_dim must be > 0, got {self.noise_network_hidden_dim}"
            )
        
        if self.noise_network_n_layers <= 0:
            raise ValueError(
                f"noise_network_n_layers must be > 0, got {self.noise_network_n_layers}"
            )
        
        if not (0.0 <= self.noise_network_dropout < 1.0):
            raise ValueError(
                f"noise_network_dropout must be in [0, 1), got {self.noise_network_dropout}"
            )
        
        if self.variance_type not in ["fixed", "learned"]:
            raise ValueError(
                f"variance_type must be 'fixed' or 'learned', got {self.variance_type}"
            )
        
        logger.info(
            f"DDPMBackwardConfig: latent_dim={self.latent_dim}, "
            f"n_diffusion_steps={self.n_diffusion_steps}, "
            f"n_cell_types={self.n_cell_types}, "
            f"variance_type={self.variance_type}"
        )


# ============================================================================
# TIMESTEP EMBEDDING
# ============================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.
    
    Converts discrete timestep indices t ∈ [0, T-1] to continuous embeddings
    using sinusoidal functions at different frequencies.
    
    Formula:
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    
    where d is the embedding dimension.
    """
    
    def __init__(self, embed_dim: int):
        """
        Initialize sinusoidal timestep embedding.
        
        Args:
            embed_dim: Dimension of timestep embedding (must be even)
        
        Raises:
            ValueError: If embed_dim is not even
        """
        super().__init__()
        
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        
        self.embed_dim = embed_dim
        
        # Pre-compute frequency factors: 1 / 10000^(2i/d)    #TODO: verify
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)
        
        logger.debug(f"Initialized SinusoidalTimestepEmbedding (dim={embed_dim})")
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal embeddings for timesteps.
        
        Args:
            t: Timestep indices, shape (batch_size,), dtype long
        
        Returns:
            Timestep embeddings, shape (batch_size, embed_dim)
        
        Raises:
            ValueError: If t has invalid shape or dtype
        """
        if len(t.shape) != 1:
            raise ValueError(f"t must be 1D tensor, got shape {t.shape}")
        
        if t.dtype not in [torch.int64, torch.long]:
            raise ValueError(f"t must have dtype long, got {t.dtype}")
        
        # t: (batch_size,) -> (batch_size, 1)
        t = t.float().unsqueeze(-1)
        
        # Compute arguments: t * freqs -> (batch_size, half_dim)
        args = t * self.freqs.unsqueeze(0)
        
        # Compute sin and cos embeddings
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embeddings  # (batch_size, embed_dim)


# ============================================================================
# NOISE PREDICTION NETWORK
# ============================================================================

class NoisePredictionNetwork(nn.Module):
    """
    Neural network ε_ψ(z^(t), t, c) that predicts noise added at timestep t.
    
    Architecture:
        1. Embed timestep t using sinusoidal encoding
        2. Embed cell type c using learned embedding
        3. Concatenate: [z^(t), timestep_embed, celltype_embed]
        4. Process through MLP
        5. Output: predicted noise ε̂ (same shape as z^(t))
    
    The network is conditioned on both timestep and cell type, allowing it to
    learn different denoising behaviors for different cell types.
    """
    
    def __init__(self, config: DDPMBackwardConfig):
        """
        Initialize noise prediction network.
        
        Args:
            config: DDPMBackwardConfig with hyperparameters
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Timestep embedding: t -> embedding
        self.timestep_embedding = SinusoidalTimestepEmbedding(
            config.timestep_embed_dim
        )
        
        # Cell type embedding: c -> embedding (learnable)
        self.celltype_embedding = nn.Embedding(
            num_embeddings=config.n_cell_types,
            embedding_dim=config.celltype_embed_dim
        )
        
        # Input dimension: latent + timestep_embed + celltype_embed
        input_dim = (
            config.latent_dim +
            config.timestep_embed_dim +
            config.celltype_embed_dim
        )
        
        # Build MLP: input -> hidden layers -> output
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, config.noise_network_hidden_dim))
        layers.append(nn.SiLU())  # Smooth activation (better than ReLU for diffusion)
        layers.append(nn.Dropout(config.noise_network_dropout))
        
        # Hidden layers
        for _ in range(config.noise_network_n_layers - 1):
            layers.append(
                nn.Linear(
                    config.noise_network_hidden_dim,
                    config.noise_network_hidden_dim
                )
            )
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(config.noise_network_dropout))
        
        # Output layer: predict noise (same dimension as latent)
        layers.append(
            nn.Linear(config.noise_network_hidden_dim, config.latent_dim)
        )
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(
            f"Initialized NoisePredictionNetwork: "
            f"input_dim={input_dim}, "
            f"hidden_dim={config.noise_network_hidden_dim}, "
            f"n_layers={config.noise_network_n_layers}"
        )
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise ε̂ = ε_ψ(z^(t), t, c).
        
        Args:
            z_t: Noisy latent at timestep t, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,), dtype long
            cell_type: Cell type indices, shape (batch_size,), dtype long
        
        Returns:
            Predicted noise ε̂, shape (batch_size, latent_dim)
        
        Raises:
            ValueError: If shapes or dtypes are invalid
        """
        batch_size = z_t.shape[0]
        
        # Validate inputs
        if z_t.shape[-1] != self.config.latent_dim:
            raise ValueError(
                f"z_t dimension mismatch. Expected {self.config.latent_dim}, "
                f"got {z_t.shape[-1]}"
            )
        
        if t.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: z_t={batch_size}, t={t.shape[0]}"
            )
        
        if cell_type.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: z_t={batch_size}, cell_type={cell_type.shape[0]}"
            )
        
        if t.dtype not in [torch.int64, torch.long]:
            raise ValueError(f"t must have dtype long, got {t.dtype}")
        
        if cell_type.dtype not in [torch.int64, torch.long]:
            raise ValueError(f"cell_type must have dtype long, got {cell_type.dtype}")
        
        # Check for invalid values
        if (t < 0).any() or (t >= self.config.n_diffusion_steps).any():
            raise ValueError(
                f"Invalid timestep indices: min={t.min()}, max={t.max()}, "
                f"valid range=[0, {self.config.n_diffusion_steps - 1}]"
            )
        
        if (cell_type < 0).any() or (cell_type >= self.config.n_cell_types).any():
            raise ValueError(
                f"Invalid cell type indices: min={cell_type.min()}, "
                f"max={cell_type.max()}, valid range=[0, {self.config.n_cell_types - 1}]"
            )
        
        # Embed timestep: (batch_size,) -> (batch_size, timestep_embed_dim)
        t_embed = self.timestep_embedding(t)
        
        # Embed cell type: (batch_size,) -> (batch_size, celltype_embed_dim)
        c_embed = self.celltype_embedding(cell_type)
        
        # Concatenate all inputs: (batch_size, total_input_dim)
        x = torch.cat([z_t, t_embed, c_embed], dim=-1)
        
        # Predict noise through MLP: (batch_size, latent_dim)
        noise_pred = self.mlp(x)
        
        return noise_pred


# ============================================================================
# VARIANCE PREDICTOR (FOR LEARNED VARIANCE)
# ============================================================================

class VariancePredictor(nn.Module):
    """
    Optional learned variance predictor.
    
    Predicts log variance log(σ_t²) for the reverse process distribution.
    Can improve sample quality compared to fixed variance.
    
    Architecture is similar to noise predictor but outputs scalar variance.
    """
    
    def __init__(self, config: DDPMBackwardConfig):
        """
        Initialize variance predictor.
        
        Args:
            config: DDPMBackwardConfig with hyperparameters
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Reuse same embedding modules as noise predictor
        self.timestep_embedding = SinusoidalTimestepEmbedding(
            config.timestep_embed_dim
        )
        
        self.celltype_embedding = nn.Embedding(
            num_embeddings=config.n_cell_types,
            embedding_dim=config.celltype_embed_dim
        )
        
        # Input dimension
        input_dim = (
            config.latent_dim +
            config.timestep_embed_dim +
            config.celltype_embed_dim
        )
        
        # Smaller MLP for variance prediction (simpler task)
        layers = []
        layers.append(nn.Linear(input_dim, config.noise_network_hidden_dim // 2))
        layers.append(nn.SiLU())
        layers.append(nn.Linear(config.noise_network_hidden_dim // 2, config.latent_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info("Initialized VariancePredictor for learned variance")
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict log variance log(σ²).
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Log variance, shape (batch_size, latent_dim)
        """
        # Embed inputs
        t_embed = self.timestep_embedding(t)
        c_embed = self.celltype_embedding(cell_type)
        
        # Concatenate and predict
        x = torch.cat([z_t, t_embed, c_embed], dim=-1)
        log_variance = self.mlp(x)
        
        return log_variance


# ============================================================================
# REVERSE PROCESS
# ============================================================================

class ReverseProcess(ABC, nn.Module):
    """
    Abstract base class for reverse (denoising) diffusion process.
    
    Defines the interface for computing mean, variance, and sampling from
    the reverse process distribution p_ψ(z^(t-1) | z^(t), c).
    """
    
    def __init__(
        self,
        config: DDPMBackwardConfig,
        variance_schedule: Dict[str, torch.Tensor]
    ):
        """
        Initialize reverse process.
        
        Args:
            config: DDPMBackwardConfig with hyperparameters
            variance_schedule: Dict with variance schedule components from forward process
                Required keys: beta, alpha, alpha_cumprod, alpha_cumprod_prev,
                             sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Register variance schedule as buffers (not trainable)
        for key, value in variance_schedule.items():
            self.register_buffer(key, value)
        
        # Validate variance schedule
        self._validate_variance_schedule()
    
    def _validate_variance_schedule(self):
        """Validate that all required variance schedule components are present."""
        
        required_keys = [
            "beta", "alpha", "alpha_cumprod", "alpha_cumprod_prev",
            "sqrt_alpha_cumprod", "sqrt_one_minus_alpha_cumprod"
        ]
        
        for key in required_keys:
            if not hasattr(self, key):
                raise ValueError(f"Missing variance schedule component: {key}")
            
            schedule_tensor = getattr(self, key)
            if schedule_tensor.shape[0] != self.config.n_diffusion_steps:
                raise ValueError(
                    f"Variance schedule {key} has wrong length: "
                    f"expected {self.config.n_diffusion_steps}, "
                    f"got {schedule_tensor.shape[0]}"
                )
    
    @abstractmethod
    def compute_mean(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor,
        noise_pred: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mean μ_ψ(z^(t), t, c) of reverse process distribution.
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
            noise_pred: Optional pre-computed noise prediction
        
        Returns:
            Mean μ, shape (batch_size, latent_dim)
        """
        pass
    
    @abstractmethod
    def compute_variance(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute variance σ² of reverse process distribution.
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Variance σ², shape (batch_size, latent_dim) or (batch_size, 1)
        """
        pass
    
    def sample_step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform single reverse step: sample z^(t-1) ~ p_ψ(z^(t-1) | z^(t), c).
        
        Args:
            z_t: Noisy latent at timestep t, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,)
            cell_type: Cell type indices, shape (batch_size,)
        
        Returns:
            Tuple of:
                - z_t_minus_1: Denoised latent at timestep t-1, shape (batch_size, latent_dim)
                - info_dict: Dictionary with intermediate values (noise_pred, mean, variance)
        """
        # Compute mean
        mean = self.compute_mean(z_t, t, cell_type)
        
        # Compute variance
        variance = self.compute_variance(z_t, t, cell_type)
        
        # Sample noise: ε' ~ N(0, I)
        noise = torch.randn_like(z_t)
        
        # Compute standard deviation: σ = √(variance)
        std = torch.sqrt(variance)
        
        # Sample: z^(t-1) = μ + σ * ε'
        # Special case: at t=0, no noise is added (deterministic)
        no_noise = (t == 0).float().reshape(-1, 1)  # (batch_size, 1)
        z_t_minus_1 = mean + (1.0 - no_noise) * std * noise
        
        # Return results and info
        info_dict = {
            "mean": mean,
            "variance": variance,
            "std": std,
            "noise_added": noise
        }
        
        return z_t_minus_1, info_dict


# ============================================================================
# UNIFIED REVERSE PROCESS (SUPERVISED & UNSUPERVISED)
# ============================================================================

class DDPMReverseProcessUnified(ReverseProcess):
    """
    Unified reverse process for both supervised and unsupervised modes.
    
    The mathematical formulation is identical for both cases:
        p_ψ(z^(t-1) | z^(t), c) = N(z^(t-1) | μ_ψ(z^(t), t, c), σ_t² I)
    
    Difference:
        - Supervised: c = c* (known/fixed cell type)
        - Unsupervised: c ~ p(c | z^(t), t) (predicted/random cell type)
    
    The reverse process itself doesn't distinguish between these cases;
    the conditioning variable c is simply passed in.
    """
    
    def __init__(
        self,
        config: DDPMBackwardConfig,
        variance_schedule: Dict[str, torch.Tensor]
    ):
        """
        Initialize unified reverse process.
        
        Args:
            config: DDPMBackwardConfig with hyperparameters
            variance_schedule: Dict with variance schedule from forward process
        """
        super().__init__(config, variance_schedule)
        
        # Initialize noise prediction network
        self.noise_network = NoisePredictionNetwork(config)
        
        # Initialize variance predictor (if learned variance)
        if config.variance_type == "learned":
            self.variance_predictor = VariancePredictor(config)
            logger.info("Using learned variance")
        else:
            self.variance_predictor = None
            logger.info("Using fixed variance (β_t)")
        
        logger.info("Initialized DDPMReverseProcessUnified")
    
    def compute_mean(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor,
        noise_pred: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mean μ_ψ(z^(t), t, c) using noise prediction formulation.
        
        Formula:
            μ_ψ(z^(t), t, c) = (1/√α_t) * (z^(t) - (β_t/√(1-ᾱ_t)) * ε̂)
        
        where ε̂ = ε_ψ(z^(t), t, c) is the predicted noise.
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
            noise_pred: Optional pre-computed noise prediction
        
        Returns:
            Mean μ, shape (batch_size, latent_dim)
        """
        # Predict noise if not provided
        if noise_pred is None:
            noise_pred = self.noise_network(z_t, t, cell_type)
        
        # Extract schedule components at timestep t
        # All have shape (batch_size,)
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t]
        
        # Reshape for broadcasting: (batch_size, 1)
        beta_t = beta_t.reshape(-1, 1)
        alpha_t = alpha_t.reshape(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.reshape(-1, 1)
        
        # Compute coefficient: β_t / √(1 - ᾱ_t)
        coef = beta_t / sqrt_one_minus_alpha_cumprod_t
        
        # Compute mean: μ = (1/√α_t) * (z^(t) - coef * ε̂)
        mean = (1.0 / torch.sqrt(alpha_t)) * (z_t - coef * noise_pred)
        
        return mean
    
    def compute_variance(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute variance σ² of reverse process distribution.
        
        Two options:
            1. Fixed: σ_t² = β_t (standard DDPM)
            2. Learned: σ_t² = exp(log_var_pred) (improved DDPM)
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Variance σ², shape (batch_size, latent_dim) or (batch_size, 1)
        """
        if self.config.variance_type == "fixed":
            # Fixed variance: σ_t² = β_t
            beta_t = self.beta[t]  # (batch_size,)
            variance = beta_t.reshape(-1, 1)  # (batch_size, 1) for broadcasting
            
        elif self.config.variance_type == "learned":
            # Learned variance: σ_t² = exp(log_var)
            log_variance = self.variance_predictor(z_t, t, cell_type)
            variance = torch.exp(log_variance)  # (batch_size, latent_dim)
        
        else:
            raise ValueError(f"Unknown variance_type: {self.config.variance_type}")
        
        return variance
    
    def log_prob(
        self,
        z_t_minus_1: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log p_ψ(z^(t-1) | z^(t), c).
        
        Useful for computing loss during training.
        
        Args:
            z_t_minus_1: Denoised latent at t-1, shape (batch_size, latent_dim)
            z_t: Noisy latent at t, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Log probability, shape (batch_size,)
        """
        # Compute mean and variance
        mean = self.compute_mean(z_t, t, cell_type)
        variance = self.compute_variance(z_t, t, cell_type)
        
        # Compute log probability: log N(z^(t-1) | μ, σ²I)
        diff = z_t_minus_1 - mean
        log_prob = -0.5 * (
            torch.sum(diff ** 2 / variance, dim=1) +
            torch.sum(torch.log(variance), dim=1) +
            self.config.latent_dim * np.log(2 * np.pi)
        )
        
        return log_prob


# ============================================================================
# SAMPLING MANAGER
# ============================================================================

class DDPMSampler(nn.Module):
    """
    Manager for sampling from reverse process.
    
    Handles the full sampling trajectory:
        z^(T) ~ N(0, I)  [start from Gaussian noise]
        For t = T, T-1, ..., 1:
            z^(t-1) ~ p_ψ(z^(t-1) | z^(t), c)  [iterative denoising]
        Return z^(0)  [final denoised latent]
    
    Supports both supervised and unsupervised sampling.
    """
    
    def __init__(
        self,
        config: DDPMBackwardConfig,
        reverse_process: DDPMReverseProcessUnified
    ):
        """
        Initialize sampler.
        
        Args:
            config: DDPMBackwardConfig with hyperparameters
            reverse_process: DDPMReverseProcessUnified instance
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        self.reverse_process = reverse_process
        
        logger.info(
            f"Initialized DDPMSampler with {config.n_diffusion_steps} steps"
        )
    
    def sample(
        self,
        n_samples: int,
        cell_type: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, list]]]:
        """
        Sample from reverse process: z^(T) → z^(T-1) → ... → z^(0).
        
        Args:
            n_samples: Number of samples to generate
            cell_type: Cell type for each sample, shape (n_samples,), dtype long
            return_trajectory: If True, return full trajectory of latents
        
        Returns:
            Tuple of:
                - z_0: Final denoised latent, shape (n_samples, latent_dim)
                - trajectory: Optional dict with keys:
                    - "z": List of latents at each timestep (length T+1)
                    - "noise_pred": List of predicted noise at each timestep
        
        Raises:
            ValueError: If inputs are invalid
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {n_samples}")
        
        if cell_type.shape[0] != n_samples:
            raise ValueError(
                f"cell_type batch size mismatch: "
                f"expected {n_samples}, got {cell_type.shape[0]}"
            )
        
        if cell_type.dtype not in [torch.int64, torch.long]:
            raise ValueError(f"cell_type must have dtype long, got {cell_type.dtype}")
        
        # Initialize trajectory storage (if requested)
        trajectory = {"z": [], "noise_pred": []} if return_trajectory else None
        
        # Start from Gaussian noise: z^(T) ~ N(0, I)
        z_t = torch.randn(
            n_samples,
            self.config.latent_dim,
            device=self.device
        )
        
        if return_trajectory:
            trajectory["z"].append(z_t.clone())
        
        logger.debug(
            f"Sampling: Starting from z^(T) ~ N(0, I), shape={z_t.shape}"
        )
        
        # Iteratively denoise: t = T, T-1, ..., 1
        for step_idx in range(self.config.n_diffusion_steps):
            # Current timestep
            t_current = self.config.n_diffusion_steps - 1 - step_idx
            
            # Create timestep tensor: (n_samples,)
            t = torch.full(
                (n_samples,),
                t_current,
                dtype=torch.long,
                device=self.device
            )
            
            # Perform reverse step: z^(t) → z^(t-1)
            z_t, info_dict = self.reverse_process.sample_step(z_t, t, cell_type)
            
            # Store trajectory
            if return_trajectory:
                trajectory["z"].append(z_t.clone())
                trajectory["noise_pred"].append(info_dict.get("noise_pred"))
            
            # Log progress periodically
            if (step_idx + 1) % 100 == 0 or step_idx == 0:
                logger.debug(
                    f"Sampling: Step {step_idx + 1}/{self.config.n_diffusion_steps}, "
                    f"t={t_current}, z_norm={z_t.norm(dim=1).mean():.4f}"
                )
        
        logger.info(
            f"Sampling complete: Generated {n_samples} samples, "
            f"final z^(0) norm={z_t.norm(dim=1).mean():.4f}"
        )
        
        return z_t, trajectory
    
    def sample_with_cell_type_inference(
        self,
        n_samples: int,
        cell_type_classifier: nn.Module,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, list]]]:
        """
        Sample with cell type inference (unsupervised mode).
        
        In unsupervised mode, cell type c is not known. We can:
            1. Sample c ~ p(c) from prior
            2. Infer c from intermediate latent using classifier
        
        This method implements option 2: infer cell type from z^(T) or z^(T/2).
        
        Args:
            n_samples: Number of samples
            cell_type_classifier: Neural network that predicts p(c | z)
            return_trajectory: If True, return full trajectory
        
        Returns:
            Tuple of:
                - z_0: Final denoised latent, shape (n_samples, latent_dim)
                - predicted_cell_types: Inferred cell types, shape (n_samples,)
                - trajectory: Optional trajectory dict
        """
        # Start from Gaussian noise: z^(T) ~ N(0, I)
        z_T = torch.randn(
            n_samples,
            self.config.latent_dim,
            device=self.device
        )
        
        # Infer cell type from initial noise (or partially denoised)
        with torch.no_grad():
            # Classifier predicts: p(c | z^(T))
            cell_type_logits = cell_type_classifier(z_T)
            predicted_cell_types = torch.argmax(cell_type_logits, dim=-1)
        
        logger.info(
            f"Unsupervised sampling: Inferred cell types from z^(T), "
            f"unique types: {predicted_cell_types.unique().tolist()}"
        )
        
        # Sample conditioned on predicted cell types
        z_0, trajectory = self.sample(
            n_samples=n_samples,
            cell_type=predicted_cell_types,
            return_trajectory=return_trajectory
        )
        
        return z_0, predicted_cell_types, trajectory


# ============================================================================
# UNIFIED REVERSE PROCESS MANAGER
# ============================================================================

class ReverseProcessManager(nn.Module):
    """
    Unified manager for reverse diffusion process.
    
    Provides a clean interface for:
        1. Reverse process initialization
        2. Single-step denoising
        3. Full trajectory sampling
        4. Log probability computation
    
    Works seamlessly for both supervised and unsupervised modes.
    """
    
    def __init__(
        self,
        config: DDPMBackwardConfig,
        variance_schedule: Dict[str, torch.Tensor]
    ):
        """
        Initialize reverse process manager.
        
        Args:
            config: DDPMBackwardConfig with hyperparameters
            variance_schedule: Dict with variance schedule from forward process
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize reverse process
        self.reverse_process = DDPMReverseProcessUnified(
            config=config,
            variance_schedule=variance_schedule
        )
        
        # Initialize sampler
        self.sampler = DDPMSampler(
            config=config,
            reverse_process=self.reverse_process
        )
        
        logger.info("Initialized ReverseProcessManager")
    
    def single_step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform single reverse step: z^(t) → z^(t-1).
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Tuple of (z_t_minus_1, info_dict)
        """
        return self.reverse_process.sample_step(z_t, t, cell_type)
    
    def sample(
        self,
        n_samples: int,
        cell_type: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, list]]]:
        """
        Sample full trajectory: z^(T) → ... → z^(0).
        
        Args:
            n_samples: Number of samples
            cell_type: Cell types, shape (n_samples,)
            return_trajectory: If True, return full trajectory
        
        Returns:
            Tuple of (z_0, trajectory)
        """
        return self.sampler.sample(n_samples, cell_type, return_trajectory)
    
    def sample_unsupervised(
        self,
        n_samples: int,
        cell_type_classifier: nn.Module,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, list]]]:
        """
        Sample with cell type inference (unsupervised).
        
        Args:
            n_samples: Number of samples
            cell_type_classifier: Cell type classifier network
            return_trajectory: If True, return trajectory
        
        Returns:
            Tuple of (z_0, predicted_cell_types, trajectory)
        """
        return self.sampler.sample_with_cell_type_inference(
            n_samples, cell_type_classifier, return_trajectory
        )
    
    def predict_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise ε̂ = ε_ψ(z^(t), t, c).
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Predicted noise, shape (batch_size, latent_dim)
        """
        return self.reverse_process.noise_network(z_t, t, cell_type)
    
    def compute_log_prob(
        self,
        z_t_minus_1: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log p_ψ(z^(t-1) | z^(t), c).
        
        Args:
            z_t_minus_1: Denoised latent, shape (batch_size, latent_dim)
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Log probability, shape (batch_size,)
        """
        return self.reverse_process.log_prob(z_t_minus_1, z_t, t, cell_type)
    
    def log_info(self):
        """Log detailed configuration information."""
        
        logger.info("=" * 70)
        logger.info("ReverseProcessManager Configuration")
        logger.info("=" * 70)
        logger.info(f"Latent dimension: {self.config.latent_dim}")
        logger.info(f"Diffusion steps: {self.config.n_diffusion_steps}")
        logger.info(f"Cell types: {self.config.n_cell_types}")
        logger.info(f"Variance type: {self.config.variance_type}")
        logger.info(f"Noise network hidden dim: {self.config.noise_network_hidden_dim}")
        logger.info(f"Noise network layers: {self.config.noise_network_n_layers}")
        logger.info(f"Timestep embed dim: {self.config.timestep_embed_dim}")
        logger.info(f"Cell type embed dim: {self.config.celltype_embed_dim}")
        logger.info("=" * 70)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_reverse_process_manager(
    latent_dim: int,
    n_diffusion_steps: int,
    n_cell_types: int,
    variance_schedule: Dict[str, torch.Tensor],
    variance_type: str = "fixed",
    noise_network_hidden_dim: int = 256,
    noise_network_n_layers: int = 3,
    device: str = "cpu"
) -> ReverseProcessManager:
    """
    Factory function to create reverse process manager with sensible defaults.
    
    Args:
        latent_dim: Dimension of latent space
        n_diffusion_steps: Number of diffusion steps
        n_cell_types: Number of cell types
        variance_schedule: Variance schedule from forward process
        variance_type: "fixed" or "learned"
        noise_network_hidden_dim: Hidden dimension for noise network
        noise_network_n_layers: Number of layers in noise network
        device: Device to use
    
    Returns:
        ReverseProcessManager instance
    """
    config = DDPMBackwardConfig(
        latent_dim=latent_dim,
        n_diffusion_steps=n_diffusion_steps,
        n_cell_types=n_cell_types,
        variance_type=variance_type,
        noise_network_hidden_dim=noise_network_hidden_dim,
        noise_network_n_layers=noise_network_n_layers,
        device=device
    )
    
    manager = ReverseProcessManager(
        config=config,
        variance_schedule=variance_schedule
    )
    
    return manager


def validate_reverse_inputs(
    z_t: torch.Tensor,
    t: torch.Tensor,
    cell_type: torch.Tensor,
    config: DDPMBackwardConfig
) -> None:
    """
    Validate inputs for reverse process.
    
    Args:
        z_t: Noisy latent
        t: Timesteps
        cell_type: Cell types
        config: Configuration
    
    Raises:
        ValueError: If validation fails
    """
    batch_size = z_t.shape[0]
    
    if z_t.shape[-1] != config.latent_dim:
        raise ValueError(
            f"Latent dimension mismatch: expected {config.latent_dim}, "
            f"got {z_t.shape[-1]}"
        )
    
    if t.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: z_t={batch_size}, t={t.shape[0]}"
        )
    
    if cell_type.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: z_t={batch_size}, cell_type={cell_type.shape[0]}"
        )
    
    if (t < 0).any() or (t >= config.n_diffusion_steps).any():
        raise ValueError(
            f"Invalid timesteps: min={t.min()}, max={t.max()}, "
            f"valid range=[0, {config.n_diffusion_steps - 1}]"
        )
    
    if (cell_type < 0).any() or (cell_type >= config.n_cell_types).any():
        raise ValueError(
            f"Invalid cell types: min={cell_type.min()}, max={cell_type.max()}, "
            f"valid range=[0, {config.n_cell_types - 1}]"
        )
    
    if z_t.isnan().any() or z_t.isinf().any():
        raise ValueError("z_t contains NaN or Inf values")
    
    if t.isnan().any() or t.isinf().any():
        raise ValueError("t contains NaN or Inf values")
    
    if cell_type.isnan().any() or cell_type.isinf().any():
        raise ValueError("cell_type contains NaN or Inf values")
