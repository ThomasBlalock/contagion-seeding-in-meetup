# %%
import numpy as np
import random
import math


class RSCGenerator:
    def __init__(self, k_avg, k_delta_avg, N=2000):
        self.N = N
        self.k_avg = k_avg
        self.k_delta_avg = k_delta_avg

        self.p_1 = (k_avg - 2*k_delta_avg) / (N - 1 - 2*k_delta_avg) 
        self.p_delta = (2*k_delta_avg) / ((N - 1)*(N - 2))

        self.links = [[] for _ in range(N)] # [[3,1,0],[],[0,6,92],...]
        self.triangles = [[] for _ in range(N)] # [[(3,1),)(7,32)],[],[(6,92)],...]

    def generate(self, seed=None):
        """Generates the topology using an optimized sampling approach"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._generate_1_simplices()
        self._generate_2_simplices()
        return self.links, self.triangles
        
    def _generate_1_simplices(self):
        """edge sampling from binomial"""
        print(f"Sampling edges with p_1 = {self.p_1:.8f}")
        num_edges = np.random.binomial(math.comb(self.N, 2), self.p_1)
        added_edges = set()
        
        # Rejection sampling
        node_list = range(self.N)
        while len(added_edges) < num_edges:
            i, j = random.sample(node_list, 2)
            edge = (i, j) if i < j else (j, i)
            
            if edge not in added_edges:
                added_edges.add(edge)
                self.links[i].append(j)
                self.links[j].append(i)
            
            print(f"\rEdges sampled: {len(added_edges)}/{num_edges}", end="")
        print()

    def _generate_2_simplices(self):
        """triangle hyperedge sampling from binomial"""
        print(f"Sampling triangles with p_delta = {self.p_delta:.8f}")
        num_triangles = np.random.binomial(math.comb(self.N, 3), self.p_delta)
        added_triangles = set()
        
        # Rejection sampling
        node_list = range(self.N)
        while len(added_triangles) < num_triangles:
            nodes = tuple(sorted(random.sample(node_list, 3)))
            if nodes not in added_triangles:
                added_triangles.add(nodes)
                i, j, k = nodes
                self.triangles[i].append((j, k))
                self.triangles[j].append((i, k))
                self.triangles[k].append((i, j))
            
            print(f"\rTriangles sampled: {len(added_triangles)}/{num_triangles}", end="")
        print()


class RandomSeeding:
    def __init__(self, N, rho_0):
        self.N = N
        self.rho_0 = rho_0

    def seed(self):
        num_initial_infected = int(self.N * self.rho_0)
        initial_infected = random.sample(range(self.N), num_initial_infected)
        return initial_infected


class SCMSimulator:
    def __init__(self, links, triangles, initial_infected, beta, beta_delta, mu):
        """Initializes the simulator with the given parameters.
        Parameters:
        - links: List of lists, where links[i] contains the neighbors of node i (1-simplices).
        - triangles: List of lists, where triangles[i] contains tuples (j, k) representing 2-simplices that include node i.
        - beta: Infection probability for 1-simplices.
        - beta_delta: Infection probability for 2-simplices.
        - mu: Recovery probability.
        - initial_infected: List of node indices that are initially infected.
        """
        self.links = links
        self.triangles = triangles
        self.N = len(links)

        self.beta = beta
        self.beta_delta = beta_delta
        self.mu = mu
        
        # 0 = Susceptible, 1 = Infected
        self.current_state = np.zeros(self.N, dtype=int)
        self.current_state[initial_infected] = 1
        
        self.rho_history = [np.mean(self.current_state)]
    
    def run(self, t_max):
        """Executes the simulation for t_max timesteps."""
        for t in range(t_max):
            self._step()
            if self.stable_state(): # Absorbing or steady state reached
                # Pad for output len consistency & break early
                self.rho_history.extend([
                    self.rho_history[-1]
                ] * (t_max - t - 1))
                break
                
        return self.rho_history

    def stable_state(self):
        if self.rho_history[-1] == 0.0:
            return True
        if len(self.rho_history) > 100 and np.isclose(self.rho_history[-1], self.rho_history[-100], atol=1e-5):
            return True
        return False
        

    def _step(self):
        next_state = np.copy(self.current_state)
        
        for i in range(self.N):
            if self.current_state[i] == 1: # Infected
                if random.random() < self.mu:
                    next_state[i] = 0
            else: # Susceptible
                m = self._count_infected_links(i)
                n = self._count_infected_triangles(i)
                
                # Total infection probability
                p_inf = 1.0 - ((1.0 - self.beta)**m)*((1.0 - self.beta_delta)**n)
                
                if random.random() < p_inf:
                    next_state[i] = 1
                    
        self.current_state = next_state
        self.rho_history.append(np.mean(self.current_state))

    def _count_infected_links(self, i):
        return sum(self.current_state[neighbor] for neighbor in self.links[i])

    def _count_infected_triangles(self, i):
        count = 0
        for j, k in self.triangles[i]:
            if self.current_state[j] == 1 and self.current_state[k] == 1:
                count += 1
        return count
# %%
# Example Usage

# Initialize the RSCGenerator
rsc_generator = RSCGenerator(k_avg=20, k_delta_avg=6, N=2000)
# Generate the topology
links, triangles = rsc_generator.generate(seed=42)

# Initialize the SCMSimulator
initial_infected = RandomSeeding(N=2000, rho_0=0.05).seed()
simulator = SCMSimulator(links, triangles, initial_infected, beta=0.03, beta_delta=0.1, mu=0.01)

# Run the simulation and get density (rho) history (density/rho = fraction of infected nodes)
rho_history = simulator.run(t_max=200)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(range(len(rho_history)), rho_history)
plt.xlabel('Time')
plt.ylabel('Rho')
plt.show()