"""Variance Reduction Techniques for Monte Carlo Simulation."""
import numpy as np
from typing import Callable, Tuple


class VarianceReduction:
    """Implements variance reduction techniques."""
    
    @staticmethod
    def antithetic_variates(price_func: Callable, S0: float, r: float, 
                           sigma: float, T: float, n_simulations: int, 
                           n_steps: int, payoff_func: Callable, **kwargs) -> Tuple[float, float]:
        """
        Use antithetic variates to reduce variance.
        
        Args:
            price_func: Monte Carlo pricing function
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_simulations: Number of simulations
            n_steps: Number of time steps
            payoff_func: Option payoff function
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random shocks
        Z = np.random.standard_normal((n_simulations // 2, n_steps))
        
        # Calculate price paths with original and antithetic variates
        log_returns_pos = drift + diffusion * Z
        log_returns_neg = drift + diffusion * (-Z)
        
        log_prices_pos = np.cumsum(log_returns_pos, axis=1)
        log_prices_neg = np.cumsum(log_returns_neg, axis=1)
        
        prices_pos = S0 * np.exp(log_prices_pos)
        prices_neg = S0 * np.exp(log_prices_neg)
        
        # Calculate payoffs
        payoffs_pos = payoff_func(prices_pos, **kwargs)
        payoffs_neg = payoff_func(prices_neg, **kwargs)
        
        # Combine payoffs
        all_payoffs = np.concatenate([payoffs_pos, payoffs_neg])
        
        # Discount to present value
        discount_factor = np.exp(-r * T)
        option_price = discount_factor * np.mean(all_payoffs)
        standard_error = discount_factor * np.std(all_payoffs) / np.sqrt(n_simulations)
        
        return option_price, standard_error
    
    @staticmethod
    def control_variates(price_func: Callable, S0: float, r: float, 
                        sigma: float, T: float, n_simulations: int, 
                        n_steps: int, payoff_func: Callable, 
                        control_price: float, **kwargs) -> Tuple[float, float]:
        """
        Use control variates to reduce variance.
        
        Args:
            price_func: Monte Carlo pricing function
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_simulations: Number of simulations
            n_steps: Number of time steps
            payoff_func: Option payoff function
            control_price: Analytical price of control variate
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random shocks
        Z = np.random.standard_normal((n_simulations, n_steps))
        
        # Calculate price paths
        log_returns = drift + diffusion * Z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)
        
        # Calculate payoffs
        payoffs = payoff_func(prices, **kwargs)
        
        # Use final stock price as control variate
        control_payoffs = prices[:, -1]
        
        # Calculate covariance and optimal coefficient
        cov = np.cov(payoffs, control_payoffs)[0, 1]
        var_control = np.var(control_payoffs)
        c = cov / var_control
        
        # Apply control variate adjustment
        expected_control = S0 * np.exp(r * T)
        adjusted_payoffs = payoffs - c * (control_payoffs - expected_control)
        
        # Discount to present value
        discount_factor = np.exp(-r * T)
        option_price = discount_factor * np.mean(adjusted_payoffs)
        standard_error = discount_factor * np.std(adjusted_payoffs) / np.sqrt(n_simulations)
        
        return option_price, standard_error
    
    @staticmethod
    def importance_sampling(price_func: Callable, S0: float, r: float, 
                           sigma: float, T: float, n_simulations: int, 
                           n_steps: int, payoff_func: Callable, 
                           drift_shift: float = 0.0, **kwargs) -> Tuple[float, float]:
        """
        Use importance sampling to reduce variance.
        
        Args:
            price_func: Monte Carlo pricing function
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_simulations: Number of simulations
            n_steps: Number of time steps
            payoff_func: Option payoff function
            drift_shift: Shift in drift for importance sampling
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        dt = T / n_steps
        original_drift = (r - 0.5 * sigma**2) * dt
        shifted_drift = original_drift + drift_shift * sigma * np.sqrt(dt)
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random shocks
        Z = np.random.standard_normal((n_simulations, n_steps))
        
        # Calculate price paths with shifted drift
        log_returns = shifted_drift + diffusion * Z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)
        
        # Calculate payoffs
        payoffs = payoff_func(prices, **kwargs)
        
        # Calculate likelihood ratio
        likelihood_ratio = np.exp(-drift_shift * np.sum(Z, axis=1) 
                                  - 0.5 * drift_shift**2 * n_steps)
        
        # Adjust payoffs by likelihood ratio
        weighted_payoffs = payoffs * likelihood_ratio
        
        # Discount to present value
        discount_factor = np.exp(-r * T)
        option_price = discount_factor * np.mean(weighted_payoffs)
        standard_error = discount_factor * np.std(weighted_payoffs) / np.sqrt(n_simulations)
        
        return option_price, standard_error
