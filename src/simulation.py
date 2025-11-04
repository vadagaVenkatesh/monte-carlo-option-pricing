"""Monte Carlo Simulation Engine for Option Pricing."""
import numpy as np
from typing import Callable, Optional


class MonteCarloEngine:
    """Core Monte Carlo simulation engine for option pricing."""
    
    def __init__(self, n_simulations: int = 10000, n_steps: int = 252, seed: Optional[int] = None):
        """
        Initialize Monte Carlo engine.
        
        Args:
            n_simulations: Number of price path simulations
            n_steps: Number of time steps per simulation
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_gbm(self, S0: float, r: float, sigma: float, T: float) -> np.ndarray:
        """
        Simulate stock price paths using Geometric Brownian Motion.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity (years)
            
        Returns:
            Array of simulated price paths (n_simulations x n_steps)
        """
        dt = T / self.n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random shocks
        Z = np.random.standard_normal((self.n_simulations, self.n_steps))
        
        # Calculate price paths
        log_returns = drift + diffusion * Z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)
        
        return prices
    
    def price_option(self, payoff_func: Callable, S0: float, r: float, 
                     sigma: float, T: float, **kwargs) -> tuple:
        """
        Price an option using Monte Carlo simulation.
        
        Args:
            payoff_func: Function that calculates option payoff
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            **kwargs: Additional parameters for payoff function
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        # Simulate price paths
        price_paths = self.simulate_gbm(S0, r, sigma, T)
        
        # Calculate payoffs
        payoffs = payoff_func(price_paths, **kwargs)
        
        # Discount to present value
        discount_factor = np.exp(-r * T)
        option_price = discount_factor * np.mean(payoffs)
        standard_error = discount_factor * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return option_price, standard_error
