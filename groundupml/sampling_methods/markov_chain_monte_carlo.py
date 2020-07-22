from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt


class Metropolis():
    # Based on Metropolis problem
    def __init__(self, n_islands):
        self.chain = []
        self.counts = []
        self.n_islands = n_islands
        self._initialize_chain(state=0)
        self.total_population = (self.n_islands * (self.n_islands+1)) / 2

    def simulate(self, iterations=10):
        print(self.chain.nodes)
        for _i in range(iterations):
            # Flip a coin to propose moving to next or previous island
            forward = np.random.choice([0, 1], p=[0.5, 0.5])
            # Step in that direction according to probability matrix
            random = np.random.uniform(low=0, high=1)
            print(self.chain.state, forward)
            proposal_probability = self.chain.nodes[self.chain.state][forward]
            print('Random: {}\nProb: {}'.format(random, proposal_probability))
            if random < proposal_probability:
                print('Accepted!')
                if forward:
                    self.chain.step_forward()
                else:
                    self.chain.step_backward()
            else:
                print('Denied!')
            self.counts[self.chain.state] += 1  # Increment count for new state
        # Plot
        plt.bar(np.arange(1, self.n_islands+1), self.counts)
        plt.title('# Of Times Islands Visited')
        plt.show()

    def _initialize_chain(self, state):
        # Island populations are equal to the number for the island 1..n
        island_populations = [np.float32(i+1) for i in range(self.n_islands)]

        def accept_probability(proposed, current):
            # Return 1 if proposed island population is greater than current
            # island population so we can use it as a probability (otherwise
            # we would get probabilities greater than 1)
            return min(1, island_populations[proposed] /
                       island_populations[current])

        # Generate probability matrix where each row represents the island
        # you are currently on, the first column represents the probabilities
        # of moving to the previous island, and the second column represents
        # the probabilities of moving to the next island
        probability_matrix = []
        for i in range(self.n_islands):
            row = [accept_probability(i-1, i),
                   accept_probability((i+1) % self.n_islands, i)]
            probability_matrix.append(row)
        probability_matrix = np.array(probability_matrix)

        self.chain = MarkovChain(state)
        self.chain.set_probabilities(probability_matrix)

        self.counts = np.zeros(self.n_islands)
        self.counts[self.chain.state] += 1  # Count the initial state


class MarkovChain():
    def __init__(self, state=0):
        self.state = state
        self.size = 0
        self.nodes = []

    def set_probabilities(self, new_matrix):
        # Takes in a numpy matrix and sets the markov chain transition
        # probabilities to the matrix
        self.nodes = new_matrix
        self.size = new_matrix.shape[0]

    def simulate(self, n_iterations):
        for _i in range(n_iterations):
            # Choose if the state will go to the previous state, stay the same
            # or move to the next state in the chain
            proposal_probabilities = self.nodes[self.state]
            direction = np.random.choice([-1, 0, 1], p=proposal_probabilities)
            # Move to new state
            previous = self.state
            if direction == 1:
                self.step_forward()
            elif direction == -1:
                self.step_backward()
            print(previous, '->', self.state)

    def step_forward(self):
        self.state = (self.state + 1) % self.size

    def step_backward(self):
        self.state = (self.state - 1) % self.size


if __name__ == '__main__':
    # np.random.seed(7)
    met = Metropolis(5)
    met.simulate(10000)
    # probs = np.array([[.1, .4, .5],
    #                   [.1, 0, .9],
    #                   [0, .5, .5],
    #                   [.6, .3, .1]])
    # mc = MarkovChain()
    # mc.set_probabilities(probs)
    # mc.simulate(10)
