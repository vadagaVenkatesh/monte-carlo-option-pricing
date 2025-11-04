"""Parallel Computing for Monte Carlo Simulations."""
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Callable, Tuple
from functools import partial


class ParallelMonteCarlo:
    """Parallel implementation of Monte Carlo simulation."""
    
    def __init__(self, n_processes: int = None):
        """
        Initialize parallel Monte Carlo engine.
        
        Args:
            n_processes: Number of processes to use (None = use all CPUs)
        """
        self.n_processes = n_processes or cpu_count()
    
    @staticmethod
    def _simulate_batch(params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a batch of price paths.
        
        Args:
            params: Dictionary containing simulation parameters
            
        Returns:
            Tuple of (price_paths, payoffs)
        """
        S0 = params['S0']
        r = params['r']
        sigma = params['sigma']
        T = params['T']
        n_sims = params['n_sims']
        n_steps = params['n_steps']
        payoff_func = params['payoff_func']
        seed = params.get('seed', None)
        kwargs = params.get('kwargs', {})
        
        if seed is not None:
            np.random.seed(seed)
        
        # Simulate price paths
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        Z = np.random.standard_normal((n_sims, n_steps))
        log_returns = drift + diffusion * Z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)
        
        # Calculate payoffs
        payoffs = payoff_func(prices, **kwargs)
        
        return prices, payoffs
    
    def price_option_parallel(self, payoff_func: Callable, S0: float, r: float,
                             sigma: float, T: float, n_simulations: int,
                             n_steps: int, **kwargs) -> Tuple[float, float]:
        """
        Price an option using parallel Monte Carlo simulation.
        
        Args:
            payoff_func: Function that calculates option payoff
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_simulations: Total number of simulations
            n_steps: Number of time steps per simulation
            **kwargs: Additional parameters for payoff function
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        # Split simulations across processes
        sims_per_process = n_simulations // self.n_processes
        
        # Create parameter dictionaries for each process
        params_list = []
        for i in range(self.n_processes):
            params = {
                'S0': S0,
                'r': r,
                'sigma': sigma,
                'T': T,
                'n_sims': sims_per_process,
                'n_steps': n_steps,
                'payoff_func': payoff_func,
                'seed': i * 1000,  # Different seed for each process
                'kwargs': kwargs
            }
            params_list.append(params)
        
        # Run simulations in parallel
        with Pool(processes=self.n_processes) as pool:
            results = pool.map(self._simulate_batch, params_list)
        
        # Combine results
        all_payoffs = np.concatenate([payoffs for _, payoffs in results])
        
        # Discount to present value
        discount_factor = np.exp(-r * T)
        option_price = discount_factor * np.mean(all_payoffs)
        standard_error = discount_factor * np.std(all_payoffs) / np.sqrt(len(all_payoffs))
        
        return option_price, standard_error
    
    def calculate_greek_parallel(self, greek_func: Callable, *args, **kwargs) -> float:
        """
        Calculate a Greek using parallel computation.
        
        Args:
            greek_func: Function that calculates the Greek
            *args: Positional arguments for greek_func
            **kwargs: Keyword arguments for greek_func
            
        Returns:
            Calculated Greek value
        """
        # For Greeks, we can parallelize the finite difference calculations
        return greek_func(*args, **kwargs)


class GPUMonteCarlo:
    """GPU-accelerated Monte Carlo simulation (placeholder for CuPy integration)."""
    
    def __init__(self):
        """
        Initialize GPU Monte Carlo engine.
        Note: Requires CuPy for GPU acceleration.
        """
        try:
            import cupy as cp
            self.cp = cp
            self.gpu_available = True
        except ImportError:
            self.cp = None
            self.gpu_available = False
            print("CuPy not available. Falling back to CPU computation.")
    
    def simulate_gbm_gpu(self, S0: float, r: float, sigma: float, T: float,
                        n_simulations: int, n_steps: int) -> np.ndarray:
        """
        Simulate stock price paths using GPU acceleration.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_simulations: Number of simulations
            n_steps: Number of time steps
            
        Returns:
            Array of simulated price paths
        """
        if not self.gpu_available:
            # Fall back to CPU (NumPy)
            dt = T / n_steps
            drift = (r - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)
            
            Z = np.random.standard_normal((n_simulations, n_steps))
            log_returns = drift + diffusion * Z
            log_prices = np.cumsum(log_returns, axis=1)
            prices = S0 * np.exp(log_prices)
            
            return prices
        
        # GPU implementation with CuPy
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * self.cp.sqrt(dt)
        
        Z = self.cp.random.standard_normal((n_simulations, n_steps))
        log_returns = drift + diffusion * Z
        log_prices = self.cp.cumsum(log_returns, axis=1)
        prices = S0 * self.cp.exp(log_prices)
        
        # Convert back to NumPy for compatibility
        return self.cp.asnumpy(prices)
    
    def price_option_gpu(self, payoff_func: Callable, S0: float, r: float,
                        sigma: float, T: float, n_simulations: int,
                        n_steps: int, **kwargs) -> Tuple[float, float]:
        """
        Price an option using GPU-accelerated Monte Carlo simulation.
        
        Args:
            payoff_func: Function that calculates option payoff
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_simulations: Number of simulations
            n_steps: Number of time steps
            **kwargs: Additional parameters for payoff function
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        # Simulate price paths
        price_paths = self.simulate_gbm_gpu(S0, r, sigma, T, n_simulations, n_steps)
        
        # Calculate payoffs
        payoffs = payoff_func(price_paths, **kwargs)
        
        # Discount to present value
        discount_factor = np.exp(-r * T)
        option_price = discount_factor * np.mean(payoffs)
        standard_error = discount_factor * np.std(payoffs) / np.sqrt(n_simulations)
        
        return option_price, standard_error
