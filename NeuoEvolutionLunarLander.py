"""
This is a my minimalisic solution to the Lular Lander gym environment using an evolutionaly NN aproach.
I did it for learning only, no optimization. 
It employs the NEAT libary for the genetic evolution control : https://neat-python.readthedocs.io/
The config and part of the code was used from: https://github.com/sroj/neat-openai-gym
Written in VS Code with Ipython Kernel (Interactive)
"""


#%%
#Import
import gym
import neat
from neat.parallel import ParallelEvaluator
import numpy as np
from functools import partial

#%%
#Set globals
ENVIRONMENT_NAME = 'LunarLander-v2'
CONFIG_FILENAME = 'LunarLander.neat'
MAX_GENS = 1000
REWARD_GOAL = 250
NUM_WORKERS = 11 #set to cpu core count
env = gym.make(ENVIRONMENT_NAME)


#%%
#Define Functions
def eval_single_genome(genome,genome_config):
    """
    This functions retruns a value on which the general fitness is measured.
    This function is passed to the main neat.Poplulation.run routine of neat
    In conjunction of the 
    """
    #Create the NeuralNet
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0

    #Run 1 time
    for i in range(1):
        state = env.reset()
        action = eval_network(net,state)
        done = False
        t = 0 
        while not done:
            state,reward,done,_=env.step(action)
            action = eval_network(net,state)
            total_reward += reward
            t += 1
            if done:
                break

    return total_reward


def eval_genomes_parallel(eval_single_genome, genomes, neat_config):
    """
    This is a wrapper so that eval runs in parallel it runs in parallel
    """
    parallel_evaluator = ParallelEvaluator(NUM_WORKERS, eval_function=eval_single_genome)
    parallel_evaluator.evaluate(genomes, neat_config)


def eval_network(net,net_input):
    """
    return an action (the output of the nn with has action_space as the end neurons)
    """
    activation = net.activate(net_input)
    return np.argmax(activation)


def test_genome(eval_network, net):
    """
    Very similar to eval_single_genome but with rendering and some output.
    Is only called after evolution is finished. So basically just cosmetics.
    """
    reward_goal = REWARD_GOAL
    TEST_COUNT = 10
    RENDER_TESTS = True
    print("Testing genome with target average reward of: {}".format(reward_goal))
    rewards = np.zeros(TEST_COUNT)
    for i in range(TEST_COUNT):
        print("--> Starting test episode trial {}".format(i + 1))
        observation = env.reset()
        action = eval_network(net, observation)
        done = False
        t = 0
        reward_episode = 0
        while not done:
            if RENDER_TESTS:
                env.render()
            observation, reward, done, _ = env.step(action)
            action = eval_network(net, observation)
            reward_episode += reward
            t += 1
            if done:
                print("<-- Test episode done after {} time steps with reward {}".format(t + 1, reward_episode))
                pass
        rewards[i % TEST_COUNT] = reward_episode
        if i + 1 >= TEST_COUNT:
            average_reward = np.mean(rewards)
            print("Average reward for episode {} is {}".format(i + 1, average_reward))
            if average_reward >= reward_goal:
                print("Hit the desired average reward in {} episodes".format(i + 1))
                break


#%%
#--------MAIN-------
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_FILENAME)
population = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(5))

#Run the evolution
winner = population.run(partial(eval_genomes_parallel, eval_single_genome), n=MAX_GENS)

# Display the winning genome.
net = neat.nn.FeedForwardNetwork.create(winner, config)
test_genome(eval_network, net)

#%%
