# Import libraries
import numpy as np
import pandas as pd

# Question A: Create a markov chain 
print("A. Build a markov chain of two states, 0 and 1")

# Set random seed
seed = 24601

# Initialize markov chain matrix representation
p = np.zeros((2,2))
np.random.seed(seed)
for i in range(2):
    p[i, 0] = np.random.rand()
    p[i, 1] = 1 - p[i, 0]

# Print initial matrix 
print("Original markov chain matrix:\n")
print(pd.DataFrame(p), "\n")

# Initialize convergence loop
pn0 = p.copy()
pn1 = pn0 @ pn0
n = 0 

# Question C: Convergence 
print("""C. Calculate p^n values until convergence at: 
      max(abs(pn1[0,0] - pn0[0,0]), 
      abs(pn1[1,1] - pn0[1,1]))
      """)

# Loop until convergence
while max(abs(pn1[0,0] - pn0[0,0]),
          abs(pn1[1,1] - pn0[1,1])) >= 0.001:
    n += 1  
    pn0 = pn1.copy()
    pn1 = pn0 @ pn0

# Find the convergent state
state = np.argmax(pn1.sum(axis=0))

# Print final matrix 
print(f"Convergence at p ^ {n}:\n")
print(pd.DataFrame(pn1), "\n")
print(f"Convergent state: {state}\n")

# Question D: Markov chain simulation
print("""D. Run a simulation of 2000 steps, 
      and calculate proportion of state 0 out of all steps
      """)

def run_simulation(p: np.ndarray, steps: int, count_states=False, return_state=False):
    """
    Run a simulation of a markov chain with a given matrix

    Args:
        p (np.ndarray): matrix representation of the markov chain
        steps (int): number of steps for the simulation

    Returns:
        depends on return conditions, a combination of:
            float: proportion of state 0 in the simulation
            int: last state in the simulation
    """
    state = 0
    other_state = 1 * (not state) # Binary flip between 1 and 0
    step = 0
    
    if count_states: 
        count_state_0 = 0
    
    # Start main loop
    while step < steps:
        
        # Count the number of times the state variable was 0
        if count_states:
            count_state_0 += 1 if state == 0 else 0
        
        # Check for next state
        change_probability = p[state, other_state]
        if np.random.rand() < change_probability:
            continue
        else:
            state = 1 * (not state) # Binary flip between 1 and 0
        
        # Advance the loop
        step += 1 
    
    # Conditional return values
    if count_states and return_state:
        return state, count_state_0 / steps
    elif count_states and not return_state:
        return count_state_0 / steps
    elif not count_states and return_state:
            return state

result = run_simulation(p=p, steps=2000, count_states=True)
print(f"Proportion of state = 0 during simulation: \n{result}\n")

# Question E: Markov chain many simulations 
print("""E. Run 2000 simulations of 2000 steps, 
      count the proportion of simulations of which 0 was the last step
      """)

def run_many_simulations(p: np.ndarray, steps: int, n_simulations: int) -> float:
    """
    Run simulations of a markov chain with a given matrix

    Args:
        p (np.ndarray): matrix representation of the markov chain
        steps (int): number of steps for a simulation
        n_simulations (int): number of simulations to run

    Returns:
        float: proportion of simulations that ended with step 0 out of all simulations
    """
    n = 0
    while n < n_simulations:
        result = run_simulation(p=p, steps=steps, return_state=True)
        if result == 0: 
            n += 1

    return n / n_simulations

result = run_many_simulations(p=p, steps=2000, n_simulations=2000)
print(f"Proportion of simulations that ended with step 0 out of all simulations: \n{result / 2000}")
