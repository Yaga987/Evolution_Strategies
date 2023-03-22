import numpy as np

class EvolutionStrategy:
    def __init__(self, population_size, sigma, learning_rate, num_params):
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_params = num_params
        self.theta = np.random.randn(num_params)
        self.best_reward = -np.inf
        self.best_theta = np.zeros(num_params)
        self.reward_history = []
        self.theta_history = []
    
    def get_reward(self, theta):
        # Placeholder function - replace with your own problem
        return np.sum(theta)
    
    def get_rewards(self, population):
        rewards = np.zeros(self.population_size)
        for i in range(self.population_size):
            rewards[i] = self.get_reward(population[i])
        return rewards
    
    def update_theta(self, rewards, population):
        # Sort population and rewards in decreasing order
        sorted_indices = np.argsort(-rewards)
        rewards = rewards[sorted_indices]
        population = population[sorted_indices]
        
        # Record best reward and theta
        if rewards[0] > self.best_reward:
            self.best_reward = rewards[0]
            self.best_theta = population[0]
        
        # Calculate standard deviation of rewards
        std_reward = np.std(rewards)
        if std_reward == 0:
            std_reward = 1
        
        # Update theta
        for i in range(self.num_params):
            self.theta[i] += (self.learning_rate / (self.population_size * self.sigma)) * np.sum((rewards - np.mean(rewards)) * population[:,i])
        
        # Record theta and reward history
        self.theta_history.append(self.theta)
        self.reward_history.append(self.best_reward)
    
    def run(self, num_iterations):
        for i in range(num_iterations):
            # Generate population
            population = np.random.randn(self.population_size, self.num_params) * self.sigma + self.theta
            
            # Evaluate population
            rewards = self.get_rewards(population)
            
            # Update theta
            self.update_theta(rewards, population)
            
            # Print progress
            print(f"Iteration {i}: Best reward = {self.best_reward}")

def get_reward(theta):
    return -theta[0]**2

es = EvolutionStrategy(population_size=50, sigma=0.1, learning_rate=0.03, num_params=1)
es.run(num_iterations=100)