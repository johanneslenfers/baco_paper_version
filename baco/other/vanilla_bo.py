import torch
import numpy as np
from typing import Callable, Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
from types import SimpleNamespace  # to simulate opentuners argparse

# BoTorch imports
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann

from baco.param.space import Space

# GPyTorch imports
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior


@dataclass
class OptimizationConfig:
    """Configuration for the Bayesian optimizer"""
    n_dims: int
    n_initial_points: int = 10
    n_iterations: int = 50
    failure_penalty: float = 1000.0  # Large positive value for minimization
    acquisition_function: str = "ei"  # "ei" or "ucb"
    ucb_beta: float = 2.0
    random_seed: Optional[int] = None
    verbose: bool = True


class BayesianOptimizer:
    """
    Bayesian optimizer using Gaussian Processes for compiler optimization.
    
    Features:
    - Matern kernel with ARD (Automatic Relevance Determination)
    - Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition
    - Robust handling of function failures
    - Designed for [0,1]^n continuous optimization spaces
    - MINIMIZES the objective function
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.n_dims = config.n_dims
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        # Storage for observations
        self.X = torch.empty(0, self.n_dims, dtype=torch.float64)
        self.Y = torch.empty(0, 1, dtype=torch.float64)
        
        # Best observed values (for minimization)
        self.best_x = None
        self.best_y = float('inf')  # Start with infinity for minimization
        self.iteration = 0
        
        # History tracking
        self.history = {
            'x': [],
            'y': [],
            'iteration': [],
            'acquisition_value': [],
            'is_failure': []
        }
    
    def _create_gp_model(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        """Create a Gaussian Process model with Matern kernel and ARD"""
        
        # Standardize targets for better numerical stability
        Y_standardized = standardize(Y)
        
        # Create GP with Matern kernel and ARD
        # Matern 2.5 kernel is good for compiler optimization (allows sharp transitions)
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.n_dims,  # ARD: different lengthscales per dimension
                lengthscale_prior=GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=GammaPrior(2.0, 0.15)
        )
        
        model = SingleTaskGP(
            X, 
            Y_standardized,
            covar_module=covar_module
        )
        
        return model
    
    def _get_acquisition_function(self, model, best_f: float):
        """Get the specified acquisition function"""
        
        if self.config.acquisition_function.lower() == "ei":
            # For minimization, we want to improve below the current best
            return ExpectedImprovement(model=model, best_f=best_f, maximize=False)
        elif self.config.acquisition_function.lower() == "ucb":
            # For minimization, use Lower Confidence Bound (negative UCB)
            return UpperConfidenceBound(model=model, beta=self.config.ucb_beta, maximize=False)
        else:
            raise ValueError(f"Unknown acquisition function: {self.config.acquisition_function}")
    
    def _optimize_acquisition(self, acq_func) -> Tuple[torch.Tensor, float]:
        """Optimize the acquisition function to find next candidate"""
        
        bounds = torch.stack([
            torch.zeros(self.n_dims, dtype=torch.float64),
            torch.ones(self.n_dims, dtype=torch.float64)
        ])
        
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=512,
        )
        
        return candidate.squeeze(0), acq_value.item()
    
    def _generate_initial_points(self) -> torch.Tensor:
        """Generate initial points using Latin Hypercube Sampling"""
        from scipy.stats import qmc
        
        sampler = qmc.LatinHypercube(d=self.n_dims, seed=self.config.random_seed)
        sample = sampler.random(n=self.config.n_initial_points)
        
        return torch.from_numpy(sample).to(torch.float64)
    
    def evaluate_function(self, x: torch.Tensor, objective_func: Callable) -> float:
        """
        Evaluate the objective function and handle failures.
        
        Args:
            x: Input point in [0,1]^n
            objective_func: Function that takes a list/array and returns Optional[float]
        
        Returns:
            Performance value or failure penalty
        """
        # Convert to numpy for the objective function
        x_np = x.detach().cpu().numpy().tolist()
        
        try:
            result = objective_func(x_np)
            if result is None:
                # Function returned None (failure)
                return self.config.failure_penalty
            else:
                return float(result)
        except Exception as e:
            if self.config.verbose:
                print(f"Function evaluation failed with exception: {e}")
            return self.config.failure_penalty
    
    def add_observation(self, x: torch.Tensor, y: float):
        """Add a new observation to the dataset"""
        
        # Convert to tensors if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        y_tensor = torch.tensor([[y]], dtype=torch.float64)
        
        # Add to dataset
        self.X = torch.cat([self.X, x], dim=0)
        self.Y = torch.cat([self.Y, y_tensor], dim=0)
        
        # Update best observation (for minimization)
        if y < self.best_y:
            self.best_y = y
            self.best_x = x.clone()
        
        # Update history
        is_failure = (y >= self.config.failure_penalty - 1e-6)  # Failures are large positive values for minimization
        self.history['x'].append(x.squeeze().detach().cpu().numpy().copy())
        self.history['y'].append(y)
        self.history['iteration'].append(self.iteration)
        self.history['acquisition_value'].append(getattr(self, '_last_acq_value', 0.0))
        self.history['is_failure'].append(is_failure)
    
    def optimize(self, objective_func: Callable) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Args:
            objective_func: Function to optimize. Takes list of floats in [0,1], 
                          returns Optional[float] (None for failure)
        
        Returns:
            Dictionary with optimization results
        """
        
        if self.config.verbose:
            print(f"Starting Bayesian Optimization with {self.config.acquisition_function.upper()} (MINIMIZING)")
            print(f"Dimensions: {self.n_dims}, Initial points: {self.config.n_initial_points}")
            print(f"Total iterations: {self.config.n_iterations}")
            print("-" * 60)
        
        # Phase 1: Initial random sampling
        if len(self.X) == 0:  # Only do initial sampling if no data exists
            if self.config.verbose:
                print("Phase 1: Initial sampling...")
            
            initial_X = self._generate_initial_points()
            
            for i, x in enumerate(initial_X):
                y = self.evaluate_function(x, objective_func)
                self.add_observation(x, y)
                
                if self.config.verbose:
                    status = "FAIL" if y >= self.config.failure_penalty - 1e-6 else "OK"
                    print(f"Initial {i+1:2d}/{self.config.n_initial_points}: y={y:8.4f} [{status}]")
        
        # Phase 2: Bayesian optimization loop
        if self.config.verbose:
            print(f"\nPhase 2: Bayesian optimization...")
            print(f"Current best: y={self.best_y:.4f}")
            print("-" * 60)
        
        for iteration in range(self.config.n_iterations):
            self.iteration = len(self.X)  # Update iteration counter
            
            # Fit GP model
            model = self._create_gp_model(self.X, self.Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Get acquisition function
            best_f = self.Y.min().item()  # Best observed value for minimization (smallest value)
            acq_func = self._get_acquisition_function(model, best_f)
            
            # Optimize acquisition function
            candidate, acq_value = self._optimize_acquisition(acq_func)
            self._last_acq_value = acq_value
            
            # Evaluate candidate
            y_new = self.evaluate_function(candidate, objective_func)
            self.add_observation(candidate, y_new)
            
            if self.config.verbose:
                status = "FAIL" if y_new >= self.config.failure_penalty - 1e-6 else "OK"
                improvement = "🎉" if y_new < self.best_y else ""
                print(f"Iter {iteration+1:2d}: y={y_new:8.4f} (acq={acq_value:6.3f}) [{status}] {improvement}")
        
        # Return results
        results = {
            'best_x': self.best_x.detach().cpu().numpy() if self.best_x is not None else None,
            'best_y': self.best_y,
            'n_evaluations': len(self.X),
            'n_failures': sum(self.history['is_failure']),
            'history': self.history.copy(),
            'final_model': model if 'model' in locals() else None
        }
        
        if self.config.verbose:
            print("-" * 60)
            print(f"Optimization completed!")
            print(f"Best performance: {self.best_y:.6f} (lower is better)")
            print(f"Total evaluations: {len(self.X)}")
            print(f"Failures: {sum(self.history['is_failure'])}/{len(self.X)} ({100*sum(self.history['is_failure'])/len(self.X):.1f}%)")
        
        return results
    
    def predict(self, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions using the trained GP model.
        
        Args:
            X_test: Test points in [0,1]^n
            
        Returns:
            Tuple of (mean, variance) predictions
        """
        if len(self.X) == 0:
            raise ValueError("No observations available. Run optimization first.")
        
        model = self._create_gp_model(self.X, self.Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_test)
            mean = posterior.mean
            variance = posterior.variance
        
        return mean, variance


# Example usage and test function
def example_compiler_objective(x: List[float]) -> Optional[float]:
    """
    Example compiler optimization function for MINIMIZATION.
    Simulates optimizing tile sizes where we want to minimize execution time.
    """
    
    # Convert [0,1] to tile sizes (e.g., 1 to 128)
    tile_sizes = [int(1 + xi * 127) for xi in x]
    
    # Simulate execution time calculation (we want to MINIMIZE this)
    execution_time = 10.0  # Base execution time
    
    for i, tile_size in enumerate(tile_sizes):
        # Smaller tile sizes are generally faster (less cache pressure)
        execution_time *= (2.0 - 0.8 * tile_size / 128.0)
        
        # But gets much slower if not power of 2 (vectorization penalty)
        if not (tile_size & (tile_size - 1) == 0):  # Not power of 2
            execution_time *= 1.5
        
        # Performance cliff if product of dimensions > 8192 (cache overflow)
        if np.prod(tile_sizes) > 8192:
            execution_time *= 3.0
    
    # Add some noise
    execution_time += np.random.normal(0, 0.1)
    
    # Simulate occasional failures (5% chance)
    if np.random.random() < 0.05:
        return None
    
    return execution_time


# BaCO Framework Integration
class BayesianOptimizerShell:
    """
    BaCO integration shell for Bayesian Optimization.
    Similar to OpentunerShell but uses our custom BO implementation.
    """
    
    def __init__(self, settings, black_box_function=None):
        self.settings = settings
        self.param_space = Space(settings)
        self.black_box_function = black_box_function
        self.beginning_of_time = None
        self.output_data_file = None
        self.optimizer = None
        
        # Extract BO configuration from settings
        bo_config = settings.get("bayesian_optimization", {})
        self.config = OptimizationConfig(
            n_dims=len(settings["input_parameters"]),
            n_initial_points=bo_config.get("n_initial_points", 10),
            n_iterations=settings.get("optimization_iterations", 50),
            acquisition_function=bo_config.get("acquisition_function", "ei"),
            ucb_beta=bo_config.get("ucb_beta", 2.0),
            failure_penalty=bo_config.get("failure_penalty", 1000.0),  # Positive for minimization
            random_seed=bo_config.get("random_seed", None),
            verbose=bo_config.get("verbose", True)
        )
    
        # self.param_space = Space(settings)
        self.beginning_of_time = self.param_space.current_milli_time()
        self.output_data_file = self.settings["output_data_file"]
        
        # Initialize our BO optimizer
        self.optimizer = BayesianOptimizer(self.config)
        
        # Add seed configurations if available
        if hasattr(self.param_space, 'chain_of_trees') and self.param_space.conditional_space:
            seed_configs = self.get_seed_configurations()
            for seed_config in seed_configs:
                # Convert seed to [0,1] tensor
                seed_tensor = torch.tensor([seed_config[name] for name in self.param_space.parameter_names], 
                                         dtype=torch.float64)
                # Evaluate seed configuration
                y_value = self.evaluate_configuration(seed_tensor)
                self.optimizer.add_observation(seed_tensor, y_value)
    
    def get_seed_configurations(self):
        """Get seed configurations similar to OpentunerShell"""
        if not self.param_space.conditional_space:
            return []
            
        cfg = self.param_space.get_default_configuration()
        if cfg is None or not self.param_space.evaluate(cfg)[0]:
            return []
        
        cfg = cfg[0]
        configuration = {}
        parameter_idx = 0
        
        for tree in self.param_space.chain_of_trees.trees:
            node = tree.root
            while node.children:
                n_children = len(node.children)
                # Find matching child
                matching_children = [(i, n) for i, n in enumerate(node.children) 
                                   if n.value == cfg[self.param_space.chain_of_trees.cot_order[parameter_idx]]]
                if matching_children:
                    child_index, node = matching_children[0]
                    configuration[node.parameter_name] = child_index / n_children
                    parameter_idx += 1
                else:
                    break
        
        return [configuration]
    
    def evaluate_configuration(self, config_tensor):
        """Evaluate a configuration using BaCO's infrastructure"""
        if self.param_space.conditional_space:
            # Revert embedding for conditional spaces
            configuration = self.revert_embedding(config_tensor)
            data_array = self.param_space.run_configurations(
                configuration.unsqueeze(0),
                self.beginning_of_time,
                settings=self.settings,
                black_box_function=self.black_box_function,
            )
        else:
            # Direct evaluation for non-conditional spaces
            data_array = self.param_space.run_configurations(
                config_tensor.unsqueeze(0),
                self.beginning_of_time,
                self.settings,
                black_box_function=self.black_box_function,
            )
        
        y_value = data_array.metrics_array[0][0].item()
        
        # Handle feasibility constraints (for minimization, failures get large positive penalty)
        if (hasattr(self.param_space, 'enable_feasible_predictor') and 
            self.param_space.enable_feasible_predictor and 
            not data_array.feasible_array[0].item()):
            return self.config.failure_penalty
        
        return y_value
    
    def revert_embedding(self, cfg_tensor):
        """Revert [0,1] embedding to original configuration (from OpentunerShell)"""
        parameters = self.param_space.parameters
        parameter_idx = 0
        partial_configurations = []
        
        # Convert tensor to dict for compatibility
        cfg = {param.name: cfg_tensor[i].item() for i, param in enumerate(parameters)}
        
        for tree in self.param_space.chain_of_trees.trees:
            node = tree.root
            while node.children:
                param_name = parameters[parameter_idx].name
                child_idx = int(np.floor((1 - 1e-6) * cfg[param_name] * len(node.children)))
                node = node.children[child_idx]
                parameter_idx += 1
            partial_configurations.append(node.get_partial_configuration())
        
        configuration = self.param_space.chain_of_trees.to_original_order(torch.cat(partial_configurations))
        return configuration
    
    def optimize(self):
        """Run Bayesian optimization using BaCO's infrastructure"""
        def baco_objective_wrapper(x_list):
            """Wrapper to convert our BO interface to BaCO's format"""
            x_tensor = torch.tensor(x_list, dtype=torch.float64)
            return self.evaluate_configuration(x_tensor)
        
        # Run optimization
        results = self.optimizer.optimize(baco_objective_wrapper)
        
        return results


def create_bayesian_optimizer_args(settings):
    """Create args namespace for BaCO compatibility (similar to create_namespace in OpentunerShell)"""
    args = SimpleNamespace()
    args.settings = settings
    args.black_box_function = None  # Will be set later
    
    # Time budget
    if settings["time_budget"] != -1:
        args.stop_after = settings["time_budget"]
    else:
        args.stop_after = float("inf")
    
    # Maximum number of evaluations
    args.test_limit = (
        settings["optimization_iterations"] +
        settings["design_of_experiment"]["number_of_samples"]
    )
    
    return args


if __name__ == "__main__":
    # Example usage
    print("Bayesian Optimizer for Compiler Tuning")
    print("=" * 50)
    
    # Configuration
    config = OptimizationConfig(
        n_dims=3,  # 3D tile size optimization
        n_initial_points=8,
        n_iterations=25,
        acquisition_function="ei",  # or "ucb"
        random_seed=42,
        verbose=True
    )
    
    # Create optimizer
    optimizer = BayesianOptimizer(config)
    
    # Run optimization
    results = optimizer.optimize(example_compiler_objective)
    
    print(f"\nBest configuration found: {results['best_x']}")
    print(f"Best performance: {results['best_y']:.6f}")
    
    print("\n" + "="*50)
    print("BaCO Integration Example:")
    print("Add this to your BaCO settings JSON:")
    print("""
    "bayesian_optimization": {
        "acquisition_function": "ei",
        "n_initial_points": 10,
        "ucb_beta": 2.0,
        "failure_penalty": 1000.0,
        "random_seed": 42,
        "verbose": true
    }
    """)