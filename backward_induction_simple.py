import numpy as np
import matplotlib.pyplot as plt

# ========================= Create Utility Function =========================
def crra_utility(c, gamma):
    """
    CRRA utility function: u(c) = c^(1-gamma) / (1-gamma) for gamma != 1
    or u(c) = log(c) for gamma = 1
    """
    if gamma == 1.0:
        return np.log(c)
    else:
        return (c**(1 - gamma)) / (1 - gamma)

# ========================= Create Markov Chain =========================
def create_endowment_markov(y_low, y_high, p_stay_low, p_stay_high):
    """
    Create 2-state Markov chain for endowment y
    
    Parameters:
    - y_low: low endowment value
    - y_high: high endowment value  
    - p_stay_low: probability of staying in low state
    - p_stay_high: probability of staying in high state
    
    Returns:
    - y_grid: array of endowment values [y_low, y_high]
    - Pi: 2x2 transition matrix
    """
    y_grid = np.array([y_low, y_high])
    
    # Transition matrix: Pi[i,j] = P(y_j tomorrow | y_i today)
    Pi = np.array([
        [p_stay_low, 1 - p_stay_low],      # From low state
        [1 - p_stay_high, p_stay_high]      # From high state
    ])
    
    return y_grid, Pi

# ========================= Solve the stationary distribution =========================
def stationary_distribution(Pi):
    """Compute stationary distribution of Markov chain"""
    eigenvals, eigenvecs = np.linalg.eig(Pi.T)
    # Find eigenvector corresponding to eigenvalue 1
    idx = np.argmin(np.abs(eigenvals - 1.0))
    pi_stat = np.real(eigenvecs[:, idx])
    pi_stat = pi_stat / np.sum(pi_stat)  # Normalize
    return pi_stat

# ========================= Interpolation and Asset Grid =========================
def interpolate(x_grid, y_values, x_new):
    """
    Simple interpolation - use numpy's interp for robustness
    """
    return float(np.interp(x_new, x_grid, y_values))

# ========================= Discretize Asset Grid =========================
def discretize_assets(amin, amax, n_a):
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = np.log(1 + np.log(1 + amax - amin))
    
    # make uniform grid
    u_grid = np.linspace(0, ubar, n_a)
    
    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin + np.exp(np.exp(u_grid) - 1) - 1

# ========================= Solve Model =========================
def solve_endowment_model(y_low, y_high, p_stay_low, p_stay_high, gamma,
                          beta, r, W_min, W_max, n_wealth, T):
    """
    Solve simple endowment economy with CRRA utility
    
    Budget constraint: W' = (W + y - c)(1 + r)
    Utility: u(c) = c^(1-gamma)/(1-gamma)
    
    Parameters:
    - y_low, y_high: low and high endowment values
    - p_stay_low, p_stay_high: persistence probabilities
    - gamma: risk aversion parameter
    - beta: discount factor
    - r: interest rate (constant)
    - W_min, W_max: wealth grid bounds
    - n_wealth: number of wealth grid points
    - T: time horizon
    
    Returns:
    - V: value function [n_states, n_wealth]
    - c_policy: consumption policy [n_states, n_wealth]
    - y_grid: endowment grid
    - Pi: transition matrix
    - W_grid: wealth grid
    """
    
    # Create Markov chain and grids
    y_grid, Pi = create_endowment_markov(y_low, y_high, p_stay_low, p_stay_high)
    W_grid = discretize_assets(W_min, W_max, n_wealth)
    n_states = len(y_grid)
    
    # Initialize value function and policy
    V = np.zeros((n_states, n_wealth))
    c_policy = np.zeros((n_states, n_wealth))
    
    # Terminal condition: V_T = 0 (or could use some other condition)
    # V is already initialized to zeros
    
    # Backward iteration
    for t in range(T-1, -1, -1):
        V_new = np.zeros((n_states, n_wealth))
        c_new = np.zeros((n_states, n_wealth))
        
        for i_state in range(n_states):
            y_today = y_grid[i_state]
            
            for i_w, W_today in enumerate(W_grid):
                
                # Define objective function for this (state, wealth) pair
                def objective(c):
                    if c <= 0 or c > W_today + y_today:
                        return -1e10  # Infeasible consumption
                    
                    # Next period wealth
                    W_next = (W_today + y_today - c) * (1 + r)
                    
                    if W_next < W_min:
                        return -1e10  # Wealth below minimum
                    
                    # Current utility
                    u_current = crra_utility(c, gamma)
                    
                    # Expected continuation value
                    if W_next > W_max:
                        # Extrapolate using last available value
                        EV_next = sum(Pi[i_state, j] * V[j, -1] for j in range(n_states))
                    else:
                        # Interpolate
                        EV_next = 0
                        for j in range(n_states):
                            if W_next <= W_min:
                                V_interp = V[j, 0]
                            elif W_next >= W_max:
                                V_interp = V[j, -1]
                            else:
                                V_interp = interpolate(W_grid, V[j, :], W_next)
                            EV_next += Pi[i_state, j] * V_interp
                    
                    return u_current + beta * EV_next
                
                # Optimize consumption
                c_max = W_today + y_today - 1e-7  # Leave small amount for next period
                if c_max <= 0:
                    c_optimal = 1e-7
                    V_optimal = -1e10
                else:
                    # Grid search for robustness
                    c_candidates = np.linspace(1e-7, c_max, 100)
                    values = [objective(c) for c in c_candidates]
                    best_idx = np.argmax(values)
                    c_optimal = c_candidates[best_idx]
                    V_optimal = values[best_idx]
                
                V_new[i_state, i_w] = V_optimal
                c_new[i_state, i_w] = c_optimal
        
        V = V_new.copy()
        c_policy = c_new.copy()
        
        # Print progress every 10 iterations
        if t % 10 == 0:
            print(f"Iteration {T-t}/{T}")
    
    return V, c_policy, y_grid, Pi, W_grid

# ========================= Plotting Functions =========================
def plot_policy_functions(V, c_policy, y_grid, Pi, W_grid, gamma, beta, r):
    """
    Plot value function and consumption policy
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Compute stationary distribution
    pi_stat = stationary_distribution(Pi)
    
    # Plot value function
    ax = axes[0, 0]
    ax.plot(W_grid, V[0, :], 'b-', label=f'Low y = {y_grid[0]:.2f}', linewidth=2)
    ax.plot(W_grid, V[1, :], 'r-', label=f'High y = {y_grid[1]:.2f}', linewidth=2)
    ax.set_xlabel('Wealth (W)')
    ax.set_ylabel('Value Function V(W, y)')
    ax.set_title('Value Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot consumption policy
    ax = axes[0, 1]
    ax.plot(W_grid, c_policy[0, :], 'b-', label=f'Low y = {y_grid[0]:.2f}', linewidth=2)
    ax.plot(W_grid, c_policy[1, :], 'r-', label=f'High y = {y_grid[1]:.2f}', linewidth=2)
    ax.set_xlabel('Wealth (W)')
    ax.set_ylabel('Consumption c(W, y)')
    ax.set_title('Consumption Policy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot savings policy (W' vs W)
    ax = axes[1, 0]
    W_next_low = (W_grid + y_grid[0] - c_policy[0, :]) * (1 + r)
    W_next_high = (W_grid + y_grid[1] - c_policy[1, :]) * (1 + r)
    ax.plot(W_grid, W_next_low, 'b-', label=f'Low y = {y_grid[0]:.2f}', linewidth=2)
    ax.plot(W_grid, W_next_high, 'r-', label=f'High y = {y_grid[1]:.2f}', linewidth=2)
    ax.plot(W_grid, W_grid, 'k--', alpha=0.5, label='45° line')
    ax.set_xlabel('Current Wealth (W)')
    ax.set_ylabel('Next Period Wealth (W\')')
    ax.set_title('Wealth Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot marginal propensity to consume
    ax = axes[1, 1]
    mpc_low = np.gradient(c_policy[0, :], W_grid)
    mpc_high = np.gradient(c_policy[1, :], W_grid)
    ax.plot(W_grid, mpc_low, 'b-', label=f'Low y = {y_grid[0]:.2f}', linewidth=2)
    ax.plot(W_grid, mpc_high, 'r-', label=f'High y = {y_grid[1]:.2f}', linewidth=2)
    ax.set_xlabel('Wealth (W)')
    ax.set_ylabel('Marginal Propensity to Consume')
    ax.set_title(r'MPC = $\frac{dc}{dW}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('policy_functions.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# Example usage and testing
if __name__ == "__main__":
    print("Solving Simple Endowment Economy with CRRA Utility")
    print("="*55)
    
    # Solve the model
    V, c_policy, y_grid, Pi, W_grid = solve_endowment_model(
        y_low=1.0,          # Low endowment
        y_high=2.4,         # High endowment
        p_stay_low=0.7,     # Persistence in low state
        p_stay_high=0.8,    # Persistence in high state
        gamma=2.0,          # Risk aversion
        beta=0.95,          # Discount factor
        r=0.03,             # Interest rate
        W_min=-0.5,         # Minimum wealth
        W_max=40.0,         # Maximum wealth
        n_wealth=50,        # Wealth grid points
        T=300               # Time horizon
    )
    
    print("Model solved successfully!")
    print(f"Endowment states: {y_grid}")
    print(f"Transition matrix:\n{Pi}")
    
    # Compute and display stationary distribution
    pi_stat = stationary_distribution(Pi)
    print(f"Stationary distribution: {pi_stat}")
    
    print(f"\nWealth grid (first 5 points): {W_grid[:5]}")
    print(f"Wealth grid (last 5 points): {W_grid[-5:]}")
    
    print(f"\nConsumption policy in low state (first 5): {c_policy[0, :5]}")
    print(f"Consumption policy in high state (first 5): {c_policy[1, :5]}")
    
    print(f"\nConsumption policy in low state (last 5): {c_policy[0, -5:]}")
    print(f"Consumption policy in high state (last 5): {c_policy[1, -5:]}")
    
    # Plot results
    print("\nGenerating plots...")
    fig = plot_policy_functions(V, c_policy, y_grid, Pi, W_grid, 
                               gamma=2.0, beta=0.95, r=0.03)
    
    
    # Test interpolation at a few points
    print(f"\nTesting interpolation:")
    for W_test in [5.0, 10.0, 15.0]:
        if W_test >= W_grid.min() and W_test <= W_grid.max():
            c_low = interpolate(W_grid, c_policy[0, :], W_test)
            c_high = interpolate(W_grid, c_policy[1, :], W_test)
            print(f"At W = {W_test}: c_low = {c_low:.4f}, c_high = {c_high:.4f}")
