"""Unit Tests for Monte Carlo Option Pricing."""
import unittest
import numpy as np
import sys
sys.path.append('..')

from src.simulation import MonteCarloEngine
from src.options import VanillaOption, AsianOption, BarrierOption, Greeks
from src.variance_reduction import VarianceReduction
from src.parallel import ParallelMonteCarlo


class TestMonteCarloEngine(unittest.TestCase):
    """Test Monte Carlo simulation engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MonteCarloEngine(n_simulations=10000, n_steps=252, seed=42)
        self.S0 = 100
        self.K = 100
        self.r = 0.05
        self.sigma = 0.2
        self.T = 1.0
    
    def test_simulate_gbm(self):
        """Test GBM simulation."""
        prices = self.engine.simulate_gbm(self.S0, self.r, self.sigma, self.T)
        
        # Check shape
        self.assertEqual(prices.shape, (10000, 252))
        
        # Check that prices are positive
        self.assertTrue(np.all(prices > 0))
        
        # Check expected final price (approximately S0 * exp(r*T))
        expected_price = self.S0 * np.exp(self.r * self.T)
        mean_final_price = np.mean(prices[:, -1])
        
        # Allow 5% tolerance due to randomness
        self.assertAlmostEqual(mean_final_price, expected_price, delta=expected_price * 0.05)
    
    def test_price_option(self):
        """Test option pricing."""
        price, std_error = self.engine.price_option(
            payoff_func=VanillaOption.call_payoff,
            S0=self.S0, r=self.r, sigma=self.sigma, T=self.T, K=self.K
        )
        
        # Check that price is positive
        self.assertGreater(price, 0)
        
        # Check that standard error is positive
        self.assertGreater(std_error, 0)
        
        # Check that price is reasonable (between 0 and S0)
        self.assertLess(price, self.S0)


class TestVanillaOption(unittest.TestCase):
    """Test vanilla option payoffs."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MonteCarloEngine(n_simulations=10000, n_steps=252, seed=42)
        self.S0 = 100
        self.K = 100
        self.r = 0.05
        self.sigma = 0.2
        self.T = 1.0
    
    def test_call_payoff(self):
        """Test call option payoff."""
        # Generate some price paths
        price_paths = np.array([[90, 95, 100], [100, 105, 110], [110, 115, 120]])
        K = 100
        
        payoffs = VanillaOption.call_payoff(price_paths, K)
        
        # Expected payoffs: max(final_price - K, 0)
        expected = np.array([0, 10, 20])
        np.testing.assert_array_equal(payoffs, expected)
    
    def test_put_payoff(self):
        """Test put option payoff."""
        # Generate some price paths
        price_paths = np.array([[90, 95, 100], [100, 105, 110], [110, 95, 90]])
        K = 100
        
        payoffs = VanillaOption.put_payoff(price_paths, K)
        
        # Expected payoffs: max(K - final_price, 0)
        expected = np.array([0, 0, 10])
        np.testing.assert_array_equal(payoffs, expected)
    
    def test_call_put_parity(self):
        """Test put-call parity relationship."""
        call_price, _ = self.engine.price_option(
            payoff_func=VanillaOption.call_payoff,
            S0=self.S0, r=self.r, sigma=self.sigma, T=self.T, K=self.K
        )
        
        put_price, _ = self.engine.price_option(
            payoff_func=VanillaOption.put_payoff,
            S0=self.S0, r=self.r, sigma=self.sigma, T=self.T, K=self.K
        )
        
        # Put-call parity: C - P = S0 - K * exp(-r*T)
        parity_diff = call_price - put_price
        expected_diff = self.S0 - self.K * np.exp(-self.r * self.T)
        
        # Allow 1% tolerance
        self.assertAlmostEqual(parity_diff, expected_diff, delta=abs(expected_diff) * 0.01)


class TestAsianOption(unittest.TestCase):
    """Test Asian option payoffs."""
    
    def test_asian_call_payoff(self):
        """Test Asian call option payoff."""
        # Generate some price paths
        price_paths = np.array([[90, 100, 110], [100, 100, 100], [110, 100, 90]])
        K = 100
        
        payoffs = AsianOption.call_payoff(price_paths, K)
        
        # Average prices: [100, 100, 100]
        # Expected payoffs: max(avg - K, 0) = [0, 0, 0]
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(payoffs, expected)
    
    def test_asian_put_payoff(self):
        """Test Asian put option payoff."""
        # Generate some price paths
        price_paths = np.array([[90, 90, 90], [100, 100, 100], [110, 110, 110]])
        K = 100
        
        payoffs = AsianOption.put_payoff(price_paths, K)
        
        # Average prices: [90, 100, 110]
        # Expected payoffs: max(K - avg, 0) = [10, 0, 0]
        expected = np.array([10, 0, 0])
        np.testing.assert_array_equal(payoffs, expected)


class TestBarrierOption(unittest.TestCase):
    """Test barrier option payoffs."""
    
    def test_up_and_out_call(self):
        """Test up-and-out call option."""
        # Price paths that breach and don't breach barrier
        price_paths = np.array([
            [100, 105, 110],  # Doesn't breach barrier (120)
            [100, 125, 110],  # Breaches barrier
            [100, 105, 115]   # Doesn't breach barrier
        ])
        K = 100
        barrier = 120
        
        payoffs = BarrierOption.up_and_out_call(price_paths, K, barrier)
        
        # Expected: [10, 0 (knocked out), 15]
        expected = np.array([10, 0, 15])
        np.testing.assert_array_equal(payoffs, expected)
    
    def test_down_and_out_put(self):
        """Test down-and-out put option."""
        # Price paths that breach and don't breach barrier
        price_paths = np.array([
            [100, 95, 90],   # Doesn't breach barrier (80)
            [100, 75, 90],   # Breaches barrier
            [100, 85, 95]    # Doesn't breach barrier
        ])
        K = 100
        barrier = 80
        
        payoffs = BarrierOption.down_and_out_put(price_paths, K, barrier)
        
        # Expected: [10, 0 (knocked out), 5]
        expected = np.array([10, 0, 5])
        np.testing.assert_array_equal(payoffs, expected)


class TestGreeks(unittest.TestCase):
    """Test Greeks calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MonteCarloEngine(n_simulations=10000, n_steps=252, seed=42)
        self.S0 = 100
        self.K = 100
        self.r = 0.05
        self.sigma = 0.2
        self.T = 1.0
    
    def test_delta_positive(self):
        """Test that call option delta is positive."""
        def price_func(S, r=self.r, sigma=self.sigma, T=self.T, K=self.K):
            engine = MonteCarloEngine(n_simulations=5000, n_steps=252, seed=42)
            return engine.price_option(
                payoff_func=VanillaOption.call_payoff,
                S0=S, r=r, sigma=sigma, T=T, K=K
            )
        
        delta = Greeks.delta(price_func, self.S0)
        
        # Call option delta should be between 0 and 1
        self.assertGreater(delta, 0)
        self.assertLess(delta, 1)
    
    def test_gamma_positive(self):
        """Test that gamma is positive."""
        def price_func(S, r=self.r, sigma=self.sigma, T=self.T, K=self.K):
            engine = MonteCarloEngine(n_simulations=5000, n_steps=252, seed=42)
            return engine.price_option(
                payoff_func=VanillaOption.call_payoff,
                S0=S, r=r, sigma=sigma, T=T, K=K
            )
        
        gamma = Greeks.gamma(price_func, self.S0)
        
        # Gamma should be positive
        self.assertGreater(gamma, 0)
    
    def test_vega_positive(self):
        """Test that vega is positive."""
        def price_func(sigma, S0=self.S0, r=self.r, T=self.T, K=self.K):
            engine = MonteCarloEngine(n_simulations=5000, n_steps=252, seed=42)
            return engine.price_option(
                payoff_func=VanillaOption.call_payoff,
                S0=S0, r=r, sigma=sigma, T=T, K=K
            )
        
        vega = Greeks.vega(price_func, self.sigma)
        
        # Vega should be positive
        self.assertGreater(vega, 0)


class TestVarianceReduction(unittest.TestCase):
    """Test variance reduction techniques."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.S0 = 100
        self.K = 100
        self.r = 0.05
        self.sigma = 0.2
        self.T = 1.0
    
    def test_antithetic_variates(self):
        """Test antithetic variates method."""
        price, std_error = VarianceReduction.antithetic_variates(
            price_func=None,
            S0=self.S0, r=self.r, sigma=self.sigma, T=self.T,
            n_simulations=10000, n_steps=252,
            payoff_func=VanillaOption.call_payoff,
            K=self.K
        )
        
        # Check that price is positive
        self.assertGreater(price, 0)
        
        # Check that standard error is positive
        self.assertGreater(std_error, 0)
        
        # Check that price is reasonable
        self.assertLess(price, self.S0)


class TestParallelMonteCarlo(unittest.TestCase):
    """Test parallel Monte Carlo implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parallel_engine = ParallelMonteCarlo(n_processes=2)
        self.S0 = 100
        self.K = 100
        self.r = 0.05
        self.sigma = 0.2
        self.T = 1.0
    
    def test_parallel_pricing(self):
        """Test parallel option pricing."""
        price, std_error = self.parallel_engine.price_option_parallel(
            payoff_func=VanillaOption.call_payoff,
            S0=self.S0, r=self.r, sigma=self.sigma, T=self.T,
            n_simulations=10000, n_steps=252,
            K=self.K
        )
        
        # Check that price is positive
        self.assertGreater(price, 0)
        
        # Check that standard error is positive
        self.assertGreater(std_error, 0)
        
        # Check that price is reasonable
        self.assertLess(price, self.S0)


if __name__ == '__main__':
    unittest.main()
