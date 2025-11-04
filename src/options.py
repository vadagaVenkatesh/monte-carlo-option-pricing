"""Option Payoff Functions and Greeks."""
import numpy as np
from typing import Union


class VanillaOption:
    """European vanilla option payoff functions."""
    
    @staticmethod
    def call_payoff(price_paths: np.ndarray, K: float) -> np.ndarray:
        """
        Calculate European call option payoff.
        
        Args:
            price_paths: Simulated price paths (n_simulations x n_steps)
            K: Strike price
            
        Returns:
            Array of payoffs for each simulation
        """
        final_prices = price_paths[:, -1]
        return np.maximum(final_prices - K, 0)
    
    @staticmethod
    def put_payoff(price_paths: np.ndarray, K: float) -> np.ndarray:
        """
        Calculate European put option payoff.
        
        Args:
            price_paths: Simulated price paths (n_simulations x n_steps)
            K: Strike price
            
        Returns:
            Array of payoffs for each simulation
        """
        final_prices = price_paths[:, -1]
        return np.maximum(K - final_prices, 0)


class AsianOption:
    """Asian option payoff functions (average price)."""
    
    @staticmethod
    def call_payoff(price_paths: np.ndarray, K: float) -> np.ndarray:
        """
        Calculate Asian call option payoff (arithmetic average).
        
        Args:
            price_paths: Simulated price paths (n_simulations x n_steps)
            K: Strike price
            
        Returns:
            Array of payoffs for each simulation
        """
        avg_prices = np.mean(price_paths, axis=1)
        return np.maximum(avg_prices - K, 0)
    
    @staticmethod
    def put_payoff(price_paths: np.ndarray, K: float) -> np.ndarray:
        """
        Calculate Asian put option payoff (arithmetic average).
        
        Args:
            price_paths: Simulated price paths (n_simulations x n_steps)
            K: Strike price
            
        Returns:
            Array of payoffs for each simulation
        """
        avg_prices = np.mean(price_paths, axis=1)
        return np.maximum(K - avg_prices, 0)


class BarrierOption:
    """Barrier option payoff functions."""
    
    @staticmethod
    def up_and_out_call(price_paths: np.ndarray, K: float, barrier: float) -> np.ndarray:
        """
        Calculate up-and-out barrier call option payoff.
        
        Args:
            price_paths: Simulated price paths (n_simulations x n_steps)
            K: Strike price
            barrier: Upper barrier price
            
        Returns:
            Array of payoffs for each simulation
        """
        final_prices = price_paths[:, -1]
        max_prices = np.max(price_paths, axis=1)
        
        # Option knocks out if barrier is breached
        knocked_out = max_prices >= barrier
        payoffs = np.maximum(final_prices - K, 0)
        payoffs[knocked_out] = 0
        
        return payoffs
    
    @staticmethod
    def down_and_out_put(price_paths: np.ndarray, K: float, barrier: float) -> np.ndarray:
        """
        Calculate down-and-out barrier put option payoff.
        
        Args:
            price_paths: Simulated price paths (n_simulations x n_steps)
            K: Strike price
            barrier: Lower barrier price
            
        Returns:
            Array of payoffs for each simulation
        """
        final_prices = price_paths[:, -1]
        min_prices = np.min(price_paths, axis=1)
        
        # Option knocks out if barrier is breached
        knocked_out = min_prices <= barrier
        payoffs = np.maximum(K - final_prices, 0)
        payoffs[knocked_out] = 0
        
        return payoffs


class Greeks:
    """Calculate option Greeks using finite differences."""
    
    @staticmethod
    def delta(price_func, S0: float, epsilon: float = 0.01, **kwargs) -> float:
        """
        Calculate Delta (∂V/∂S) using finite difference.
        
        Args:
            price_func: Function that prices the option
            S0: Initial stock price
            epsilon: Bump size for finite difference
            **kwargs: Additional parameters for price_func
            
        Returns:
            Delta value
        """
        price_up, _ = price_func(S0 * (1 + epsilon), **kwargs)
        price_down, _ = price_func(S0 * (1 - epsilon), **kwargs)
        
        return (price_up - price_down) / (2 * S0 * epsilon)
    
    @staticmethod
    def gamma(price_func, S0: float, epsilon: float = 0.01, **kwargs) -> float:
        """
        Calculate Gamma (∂²V/∂S²) using finite difference.
        
        Args:
            price_func: Function that prices the option
            S0: Initial stock price
            epsilon: Bump size for finite difference
            **kwargs: Additional parameters for price_func
            
        Returns:
            Gamma value
        """
        price_up, _ = price_func(S0 * (1 + epsilon), **kwargs)
        price_center, _ = price_func(S0, **kwargs)
        price_down, _ = price_func(S0 * (1 - epsilon), **kwargs)
        
        return (price_up - 2 * price_center + price_down) / (S0 * epsilon) ** 2
    
    @staticmethod
    def vega(price_func, sigma: float, epsilon: float = 0.01, **kwargs) -> float:
        """
        Calculate Vega (∂V/∂σ) using finite difference.
        
        Args:
            price_func: Function that prices the option
            sigma: Volatility
            epsilon: Bump size for finite difference
            **kwargs: Additional parameters for price_func
            
        Returns:
            Vega value
        """
        price_up, _ = price_func(sigma=sigma + epsilon, **kwargs)
        price_down, _ = price_func(sigma=sigma - epsilon, **kwargs)
        
        return (price_up - price_down) / (2 * epsilon)
