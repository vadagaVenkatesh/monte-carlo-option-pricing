"""Example Usage of Monte Carlo Option Pricing Library."""
import sys
sys.path.append('..')

from src.simulation import MonteCarloEngine
from src.options import VanillaOption, AsianOption, BarrierOption, Greeks
from src.variance_reduction import VarianceReduction
from src.parallel import ParallelMonteCarlo, GPUMonteCarlo


def example_vanilla_options():
    """Example: Price European call and put options."""
    print("\n=== Vanilla Options ===")
    
    # Market parameters
    S0 = 100      # Initial stock price
    K = 100       # Strike price
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility
    T = 1.0       # Time to maturity (1 year)
    
    # Create Monte Carlo engine
    engine = MonteCarloEngine(n_simulations=10000, n_steps=252, seed=42)
    
    # Price European call option
    call_price, call_se = engine.price_option(
        payoff_func=VanillaOption.call_payoff,
        S0=S0, r=r, sigma=sigma, T=T, K=K
    )
    
    print(f"European Call Option:")
    print(f"  Price: ${call_price:.4f}")
    print(f"  Std Error: ${call_se:.4f}")
    
    # Price European put option
    put_price, put_se = engine.price_option(
        payoff_func=VanillaOption.put_payoff,
        S0=S0, r=r, sigma=sigma, T=T, K=K
    )
    
    print(f"\nEuropean Put Option:")
    print(f"  Price: ${put_price:.4f}")
    print(f"  Std Error: ${put_se:.4f}")


def example_asian_options():
    """Example: Price Asian call and put options."""
    print("\n=== Asian Options ===")
    
    # Market parameters
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    engine = MonteCarloEngine(n_simulations=10000, n_steps=252, seed=42)
    
    # Price Asian call option
    asian_call_price, asian_call_se = engine.price_option(
        payoff_func=AsianOption.call_payoff,
        S0=S0, r=r, sigma=sigma, T=T, K=K
    )
    
    print(f"Asian Call Option:")
    print(f"  Price: ${asian_call_price:.4f}")
    print(f"  Std Error: ${asian_call_se:.4f}")
    
    # Price Asian put option
    asian_put_price, asian_put_se = engine.price_option(
        payoff_func=AsianOption.put_payoff,
        S0=S0, r=r, sigma=sigma, T=T, K=K
    )
    
    print(f"\nAsian Put Option:")
    print(f"  Price: ${asian_put_price:.4f}")
    print(f"  Std Error: ${asian_put_se:.4f}")


def example_barrier_options():
    """Example: Price barrier options."""
    print("\n=== Barrier Options ===")
    
    # Market parameters
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    barrier = 120  # Barrier level
    
    engine = MonteCarloEngine(n_simulations=10000, n_steps=252, seed=42)
    
    # Price up-and-out call option
    uoc_price, uoc_se = engine.price_option(
        payoff_func=BarrierOption.up_and_out_call,
        S0=S0, r=r, sigma=sigma, T=T, K=K, barrier=barrier
    )
    
    print(f"Up-and-Out Call Option (Barrier=${barrier}):")
    print(f"  Price: ${uoc_price:.4f}")
    print(f"  Std Error: ${uoc_se:.4f}")
    
    # Price down-and-out put option
    barrier_low = 80
    dop_price, dop_se = engine.price_option(
        payoff_func=BarrierOption.down_and_out_put,
        S0=S0, r=r, sigma=sigma, T=T, K=K, barrier=barrier_low
    )
    
    print(f"\nDown-and-Out Put Option (Barrier=${barrier_low}):")
    print(f"  Price: ${dop_price:.4f}")
    print(f"  Std Error: ${dop_se:.4f}")


def example_greeks():
    """Example: Calculate option Greeks."""
    print("\n=== Option Greeks ===")
    
    # Market parameters
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    engine = MonteCarloEngine(n_simulations=10000, n_steps=252, seed=42)
    
    # Define price function for Greeks calculation
    def price_func(S, sigma=sigma, r=r, T=T, K=K):
        return engine.price_option(
            payoff_func=VanillaOption.call_payoff,
            S0=S, r=r, sigma=sigma, T=T, K=K
        )
    
    # Calculate Delta
    delta = Greeks.delta(price_func, S0)
    print(f"Delta: {delta:.4f}")
    
    # Calculate Gamma
    gamma = Greeks.gamma(price_func, S0)
    print(f"Gamma: {gamma:.6f}")
    
    # Calculate Vega
    vega = Greeks.vega(lambda sigma: price_func(S0, sigma=sigma)[0], sigma)
    print(f"Vega: {vega:.4f}")


def example_variance_reduction():
    """Example: Use variance reduction techniques."""
    print("\n=== Variance Reduction ===")
    
    # Market parameters
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    # Standard Monte Carlo
    engine = MonteCarloEngine(n_simulations=10000, n_steps=252, seed=42)
    standard_price, standard_se = engine.price_option(
        payoff_func=VanillaOption.call_payoff,
        S0=S0, r=r, sigma=sigma, T=T, K=K
    )
    
    print(f"Standard Monte Carlo:")
    print(f"  Price: ${standard_price:.4f}")
    print(f"  Std Error: ${standard_se:.4f}")
    
    # Antithetic variates
    av_price, av_se = VarianceReduction.antithetic_variates(
        price_func=None,
        S0=S0, r=r, sigma=sigma, T=T,
        n_simulations=10000, n_steps=252,
        payoff_func=VanillaOption.call_payoff,
        K=K
    )
    
    print(f"\nAntithetic Variates:")
    print(f"  Price: ${av_price:.4f}")
    print(f"  Std Error: ${av_se:.4f}")
    print(f"  SE Reduction: {(1 - av_se/standard_se) * 100:.2f}%")


def example_parallel_computation():
    """Example: Use parallel computation."""
    print("\n=== Parallel Computation ===")
    
    # Market parameters
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    # Parallel Monte Carlo
    parallel_engine = ParallelMonteCarlo(n_processes=4)
    parallel_price, parallel_se = parallel_engine.price_option_parallel(
        payoff_func=VanillaOption.call_payoff,
        S0=S0, r=r, sigma=sigma, T=T,
        n_simulations=10000, n_steps=252,
        K=K
    )
    
    print(f"Parallel Monte Carlo (4 processes):")
    print(f"  Price: ${parallel_price:.4f}")
    print(f"  Std Error: ${parallel_se:.4f}")
    
    # GPU Monte Carlo (if available)
    gpu_engine = GPUMonteCarlo()
    if gpu_engine.gpu_available:
        gpu_price, gpu_se = gpu_engine.price_option_gpu(
            payoff_func=VanillaOption.call_payoff,
            S0=S0, r=r, sigma=sigma, T=T,
            n_simulations=10000, n_steps=252,
            K=K
        )
        
        print(f"\nGPU Monte Carlo:")
        print(f"  Price: ${gpu_price:.4f}")
        print(f"  Std Error: ${gpu_se:.4f}")
    else:
        print("\nGPU computation not available (CuPy not installed)")


if __name__ == "__main__":
    print("Monte Carlo Option Pricing - Examples")
    print("=" * 50)
    
    # Run examples
    example_vanilla_options()
    example_asian_options()
    example_barrier_options()
    example_greeks()
    example_variance_reduction()
    example_parallel_computation()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
