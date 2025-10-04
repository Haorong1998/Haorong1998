import numpy as np
import numba
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d

@numba.njit
def u(c, l, theta, eis, eta):
    x = ((1-theta)*c**(1-eta) + theta*l**(1-eta))**(1/(1-eta))
    return (x**(1 - 1/eis) - 1) / (1 - 1/eis) if eis != 1 else np.log(c)

def create_joint_markov_chain(pi_stay_low=0.05, pi_stay_high=0.90, w_low=8.0, w_high=10.0, r_low=0.025, r_high=0.05):
    """
    Create simple joint Markov chain for (w, r) with 2 states:
    State 0: (w_low, r_low)
    State 1: (w_high, r_high)
    
    Transition probabilities:
    - P(w_low, r_low | w_low, r_low) = pi_stay_low
    - P(w_high, r_high | w_low, r_low) = 1 - pi_stay_low
    - P(w_low, r_low | w_high, r_high) = 1 - pi_stay_high  
    - P(w_high, r_high | w_high, r_high) = pi_stay_high
    """
    
    # Create state grids - only 2 states now
    w_grid = np.array([w_low, w_high])
    r_grid = np.array([r_low, r_high])
    
    # Initialize 2x2 transition matrix
    Pi = np.array([
        [pi_stay_low,     1 - pi_stay_low],      # From state 0 (low, low)
        [1 - pi_stay_high, pi_stay_high]         # From state 1 (high, high)
    ])
    
    return Pi, w_grid, r_grid

def stationary_markov(Pi, tol=1E-14):
    # start with uniform distribution over all states
    n = Pi.shape[0]
    pi = np.full(n, 1/n)
    
    # update distribution using Pi until successive iterations differ by less than tol
    for _ in range(10_000):
        pi_new = Pi.T @ pi
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new

def backward_iteration_with_labor(V, Pi, W_grid, w_grid, r_grid, beta, theta, eis, eta):
    """
    Backward iteration with endogenous labor choice and budget constraint:
    W′ = [W + w(1−n) − c] exp(r′)
    
    Uses high-accuracy interpolation and optimization for better precision.
    
    Args:
        V: Value function, shape (n_states, n_wealth)
        Pi: Transition matrix, shape (n_states, n_states)  
        W_grid: Wealth grid
        w_grid: Wage in each state
        r_grid: Interest rate in each state
        beta: Discount factor
        theta: Labor preference parameter in utility
        eis: Elasticity of intertemporal substitution
        eta: Risk aversion parameter
    """
    n_states = len(w_grid)
    n_wealth = len(W_grid)
    
    # Expected value function for next period
    EV = Pi @ V
    
    # Create interpolation functions for expected value
    EV_interp = []
    for s in range(n_states):
        # Use linear interpolation with extrapolation
        def create_interp(s_idx):
            def interp_func(W_next):
                return np.interp(W_next, W_grid, EV[s_idx, :])
            return interp_func
        EV_interp.append(create_interp(s))
    
    # Initialize policy functions
    c_policy = np.zeros((n_states, n_wealth))
    l_policy = np.zeros((n_states, n_wealth))  # leisure policy
    V_new = np.zeros((n_states, n_wealth))
    
    # Solve for each state and wealth level
    for s in range(n_states):
        w = w_grid[s]
        
        for i, W in enumerate(W_grid):
            
            def objective(x):
                """
                Objective function to maximize: current utility + beta * expected continuation
                x = [l, c] where l is leisure and c is consumption
                """
                l, c = x
                
                # Strict constraints
                if l <= 1e-5 or l >= 1-1e-59:  # Leisure bounds
                    return 1e10
                if c <= 1e-5:  # Consumption must be positive
                    return 1e10
                
                # Available resources: W + wage*labor - consumption
                # Since labor = (1 - leisure), we have:
                labor = 1 - l
                available = W + w * labor
                
                if c >= available - 1e-5:  # Can't consume more than available (with small buffer)
                    return 1e10
                    
                # Savings
                savings = available - c
                if savings <= 1e-5:  # Need positive savings (with small buffer)
                    return 1e10
                
                try:
                    # Current period utility (c, leisure)
                    current_u = u(c, l, theta, eis, eta)
                    
                    # Expected continuation value
                    expected_continuation = 0.0
                    for s_next in range(n_states):
                        r_next = r_grid[s_next]
                        W_next = savings * np.exp(r_next)
                        
                        # Use interpolation function
                        V_next = EV_interp[s_next](W_next)
                        expected_continuation += Pi[s, s_next] * V_next
                    
                    # Total value (negative because minimize minimizes)
                    total_value = current_u + beta * expected_continuation
                    return -total_value
                    
                except (ValueError, RuntimeWarning, OverflowError, ZeroDivisionError):
                    return 1e10
            
            # Initial guess based on simple heuristics
            l_init = 0.3  # Initial leisure guess
            labor_init = 1 - l_init
            available_init = W + w * labor_init
            c_init = min(0.5 * available_init, available_init - 0.1)  # Conservative consumption
            
            # Ensure initial guess is feasible
            if c_init <= 1e-5:
                c_init = 1e-5
            if available_init - c_init <= 1e-5:
                c_init = available_init - 1e-5
            
            # Use scipy optimization with bounds
            try:
                
                # Set bounds: l in [1e-5, 1-(1e-5)], c > 0
                bounds = [(1e-5, 1-(1e-5)), (1e-5, None)]
                
                result = minimize(objective, x0=[l_init, c_init], 
                                method='L-BFGS-B', bounds=bounds,
                                options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 1000})
                
                if result.success and result.fun < 1e9:
                    best_l, best_c = result.x
                    # Double-check constraints
                    labor = 1 - best_l
                    available_check = W + w * labor
                    if (1e-5 <= best_l <= 1-(1e-5) and best_c > 1e-5 and 
                        best_c < available_check - 1e-5):
                        best_value = -result.fun
                    else:
                        # Fallback to grid search
                        best_l, best_c, best_value = fallback_grid_search(W, w, s, EV_interp, Pi, W_grid, r_grid, beta, theta, eis, eta)
                else:
                    # Fallback to grid search if optimization fails
                    best_l, best_c, best_value = fallback_grid_search(W, w, s, EV_interp, Pi, W_grid, r_grid, beta, theta, eis, eta)
                    
            except Exception as e:
                # Fallback to grid search if optimization fails
                best_l, best_c, best_value = fallback_grid_search(W, w, s, EV_interp, Pi, W_grid, r_grid, beta, theta, eis, eta)
            
            c_policy[s, i] = best_c
            l_policy[s, i] = best_l  
            V_new[s, i] = best_value
    
    return V_new, c_policy, l_policy


def fallback_grid_search(W, w, s, EV_interp, Pi, W_grid, r_grid, beta, theta, eis, eta):
    """
    Fallback grid search with finer grids for accuracy
    """
    n_states = len(r_grid)
    best_value = -np.inf
    best_c = 0.0
    best_l = 0.0  # leisure
    
    # Finer grid search over leisure (50 points instead of 20)
    l_grid = np.linspace(1e-5, 1-(1e-5), 50)
    
    for l in l_grid:
        # Available resources: W + wage*labor where labor = 1 - leisure
        labor = 1 - l
        available = W + w * labor
        
        # Finer grid search over consumption (50 points instead of 20)
        c_max = min(available - 1e-5, available * (1 - 1e-5))
        if c_max <= 1e-5:
            continue
            
        c_grid = np.linspace(1e-5, c_max, 50)
        
        for c in c_grid:
            # Savings
            savings = available - c
            if savings <= 0:
                continue
            
            try:
                # Current period utility (consumption, leisure)
                current_u = u(c, l, theta, eis, eta)
                
                # Expected continuation value
                expected_continuation = 0.0
                for s_next in range(n_states):
                    r_next = r_grid[s_next]
                    W_next = savings * np.exp(r_next)
                    
                    # Use interpolation function
                    V_next = EV_interp[s_next](W_next)
                    expected_continuation += Pi[s, s_next] * V_next
                
                # Total value
                total_value = current_u + beta * expected_continuation
                
                if total_value > best_value:
                    best_value = total_value
                    best_c = c
                    best_l = l
                    
            except (ValueError, RuntimeWarning):
                continue
    
    return best_l, best_c, best_value


def solve_model_with_joint_markov(beta=0.96, eis=2.0, eta=2.0, theta=0.5, 
                                 n_wealth=50, W_max=20.0, 
                                 w_low=0.8, w_high=1.2, r_low=0.02, r_high=0.06,
                                 pi_stay_low=0.05, pi_stay_high=0.90, 
                                 max_iter=1000, tol=1e-6):
    """
    Solve the consumption-saving-leisure model with joint Markov chain for wages and interest rates.
    Budget constraint: W′ = [W + w*labor − c] exp(r′) where labor = (1 - leisure)
    
    Returns:
        V: Value function
        c_policy: Consumption policy function  
        l_policy: Leisure policy function
        Pi: Transition matrix
        w_grid: Wage grid
        r_grid: Interest rate grid
        W_grid: Wealth grid
    """
    
    # Create joint Markov chain
    Pi, w_grid, r_grid = create_joint_markov_chain(
        pi_stay_low=pi_stay_low, pi_stay_high=pi_stay_high,
        w_low=w_low, w_high=w_high, r_low=r_low, r_high=r_high
    )
    
    # Create wealth grid
    W_grid = np.linspace(0.1, W_max, n_wealth)  # Start from small positive value
    
    # Initialize value function and policy functions
    n_states = len(w_grid)
    V = np.zeros((n_states, n_wealth))
    c_policy = np.zeros((n_states, n_wealth))
    l_policy = np.zeros((n_states, n_wealth))  # leisure policy
    error = float('inf')
    
    # Iterate until convergence
    for iteration in range(max_iter):
        V_new, c_policy, l_policy = backward_iteration_with_labor(
            V, Pi, W_grid, w_grid, r_grid, beta, theta, eis, eta
        )
        
        # Check convergence
        error = np.max(np.abs(V_new - V))
        if error < tol:
            print(f"Converged after {iteration} iterations with error {error:.2e}")
            break
            
        V = V_new
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, error: {error:.2e}")
    
    else:
        print(f"Warning: Did not converge after {max_iter} iterations. Error: {error:.2e}")
    
    return V, c_policy, l_policy, Pi, w_grid, r_grid, W_grid


# Example usage
if __name__ == "__main__":
    # Solve the model with your specified transition probabilities
    V, c_policy, l_policy, Pi, w_grid, r_grid, W_grid = solve_model_with_joint_markov(
        pi_stay_low=0.05,   # P(w_low, r_low | w_low, r_low) = 0.05
        pi_stay_high=0.90,  # P(w_high, r_high | w_high, r_high) = 0.90
        theta=0.5,          # Leisure preference parameter
        eta=2.0,            # Risk aversion parameter
        eis=2.0             # Elasticity of intertemporal substitution
    )
    
    print("Model solved successfully!")
    print(f"States: {len(w_grid)}")
    print(f"Wage grid: {w_grid}")
    print(f"Interest rate grid: {r_grid}")
    print(f"Transition matrix:\n{Pi}")
    
    # Show stationary distribution
    pi_stationary = stationary_markov(Pi)
    print(f"Stationary distribution: {pi_stationary}")
    
    print(f"Value function shape: {V.shape}")
    print(f"Consumption policy shape: {c_policy.shape}")
    print(f"Leisure policy shape: {l_policy.shape}")
    
    # Show some policy values
    print(f"\nWealth grid (first 5): {W_grid[:5]}")
    print(f"Consumption policy in state 0 (first 5): {c_policy[0, :5]}")
    print(f"Leisure policy in state 0 (first 5): {l_policy[0, :5]}")
    print(f"Implied labor in state 0 (first 5): {1 - l_policy[0, :5]}")