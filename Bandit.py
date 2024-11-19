"""
Run this file first, to see the log messages. Instead of the print() use the respective log level.
"""

############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, chosen_arm, reward):
        pass

    @abstractmethod
    def experiment(self, num_trials):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization:
    def plot1(self, alg1_rewards, alg2_rewards, num_trials):
        """
        Visualize the performance of each bandit: linear and log scale.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(range(num_trials), np.cumsum(alg1_rewards), label='Epsilon-Greedy')
        plt.plot(range(num_trials), np.cumsum(alg2_rewards), label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot2(self, alg1_regrets, alg2_regrets, num_trials):
        """
        Compare E-greedy and Thompson Sampling cumulative regrets.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(range(num_trials), np.cumsum(alg1_regrets), label='Epsilon-Greedy')
        plt.plot(range(num_trials), np.cumsum(alg2_regrets), label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Regret')
        plt.title('Cumulative Regret over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

#--------------------------------------#


class EpsilonGreedy(Bandit):
    def __init__(self, p):
        """
        Initialize the EpsilonGreedy algorithm.

        Args:
            p (list): List of true probabilities for each bandit arm.
        """
        self.p = p 
        self.n_arms = len(p)
        self.counts = [0] * self.n_arms  
        self.estimates = [0.0] * self.n_arms  
        self.total_reward = 0.0
        self.rewards = [] 
        self.epsilon = 1.0 
        self.t = 1 
        self.regrets = [] 
        self.best_prob = max(p)  
        self.algorithm = 'Epsilon-Greedy'
        self.data = [] 

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon})"

    def pull(self):
        """
        Decide which arm to pull based on epsilon-greedy strategy.

        Returns:
            int: Index of the chosen arm.
        """
        if np.random.rand() < self.epsilon:
            chosen_arm = np.random.randint(0, self.n_arms)
        else:
            chosen_arm = np.argmax(self.estimates)
        return chosen_arm

    def update(self, chosen_arm, reward):
        """
        Update the estimates for the chosen arm.

        Args:
            chosen_arm (int): Index of the chosen arm.
            reward (float): Observed reward.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.estimates[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.estimates[chosen_arm] = new_value

    def experiment(self, num_trials):
        """
        Run the epsilon-greedy experiment.

        Args:
            num_trials (int): Number of trials to run.
        """
        for _ in range(num_trials):
            self.epsilon = 1 / self.t  
            chosen_arm = self.pull()
            # Simulate pulling the arm
            reward = 1 if np.random.rand() < self.p[chosen_arm] else 0
            self.update(chosen_arm, reward)
            self.rewards.append(reward)
            self.total_reward += reward
            regret = self.best_prob - self.p[chosen_arm]
            self.regrets.append(regret)
            # Store data for CSV
            self.data.append({'Bandit': chosen_arm, 'Reward': reward, 'Algorithm': self.algorithm})
            self.t += 1

    def report(self):
        """
        Generate a report of the experiment.
        """
        df = pd.DataFrame(self.data)
        df.to_csv('epsilon_greedy_results.csv', index=False)
        logger.info(f"Epsilon-Greedy Cumulative Reward: {self.total_reward}")
        total_regret = sum(self.regrets)
        logger.info(f"Epsilon-Greedy Cumulative Regret: {total_regret}")


#--------------------------------------#


class ThompsonSampling(Bandit):
    def __init__(self, p):
        """
        Initialize the ThompsonSampling algorithm.

        Args:
            p (list): List of true probabilities for each bandit arm.
        """
        self.p = p  
        self.n_arms = len(p)
        self.a = [1.0] * self.n_arms 
        self.b = [1.0] * self.n_arms 
        self.total_reward = 0.0
        self.rewards = [] 
        self.t = 1 
        self.regrets = [] 
        self.best_prob = max(p)  
        self.algorithm = 'Thompson Sampling'
        self.data = [] 

    def __repr__(self):
        return f"ThompsonSampling()"

    def pull(self):
        """
        Decide which arm to pull based on Thompson Sampling strategy.

        Returns:
            int: Index of the chosen arm.
        """
        samples = [np.random.beta(self.a[i], self.b[i]) for i in range(self.n_arms)]
        chosen_arm = np.argmax(samples)
        return chosen_arm

    def update(self, chosen_arm, reward):
        """
        Update the Beta distribution parameters for the chosen arm.

        Args:
            chosen_arm (int): Index of the chosen arm.
            reward (float): Observed reward.
        """
        # Assuming reward is 0 or 1
        self.a[chosen_arm] += reward
        self.b[chosen_arm] += 1 - reward

    def experiment(self, num_trials):
        """
        Run the Thompson Sampling experiment.

        Args:
            num_trials (int): Number of trials to run.
        """
        for _ in range(num_trials):
            chosen_arm = self.pull()
            reward = 1 if np.random.rand() < self.p[chosen_arm] else 0
            self.update(chosen_arm, reward)
            self.rewards.append(reward)
            self.total_reward += reward
            regret = self.best_prob - self.p[chosen_arm]
            self.regrets.append(regret)
            # Store data for CSV
            self.data.append({'Bandit': chosen_arm, 'Reward': reward, 'Algorithm': self.algorithm})
            self.t += 1

    def report(self):
        """
        Generate a report of the experiment.
        """
        # Store data in CSV
        df = pd.DataFrame(self.data)
        df.to_csv('thompson_sampling_results.csv', index=False)
        logger.info(f"Thompson Sampling Cumulative Reward: {self.total_reward}")
        total_regret = sum(self.regrets)
        logger.info(f"Thompson Sampling Cumulative Regret: {total_regret}")



def comparison():
    # Compare the performances of the two algorithms visually.
    pass

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    logger.debug("Starting the bandit simulations...")

    p = [0.2, 0.5, 0.75, 0.9]
    NumberOfTrials = 20000

    eg = EpsilonGreedy(p)
    ts = ThompsonSampling(p)

    eg.experiment(NumberOfTrials)
    ts.experiment(NumberOfTrials)

    eg.report()
    ts.report()

    viz = Visualization()
    viz.plot1(eg.rewards, ts.rewards, NumberOfTrials)
    viz.plot2(eg.regrets, ts.regrets, NumberOfTrials)

    total_data = eg.data + ts.data
    df_total = pd.DataFrame(total_data)
    df_total.to_csv('bandit_rewards.csv', index=False)

    logger.debug("Bandit simulations completed.")
