import random

from Solver import Solver
from typing import Literal, cast
import numpy as np

DEBUG = False

INF = 10000000
GOAL = b'G'
HOLE = b'H'

class GeneticSolver(Solver):
    
    map: np.ndarray
    map_height: int
    map_width: int
    actions: list[Literal[0, 1, 2, 3]]
    cost: int = -1

    def get_position(self):
        return self.gym_env.unwrapped.s

    def __init__(self, gym_env):
        super().__init__(gym_env)
        self.map = self.gym_env.unwrapped.desc
        self.map_height, self.map_width = self.map.shape
        
        self.max_actions_per_individual = self.map_height * self.map_height

        self.random = random.Random(x=0)
        if DEBUG: print(f"{self.__class__.__name__} initialized with map:")
        if DEBUG: print("map height:", self.map_height)
        if DEBUG: print("map width:", self.map_width)
        if DEBUG: print("initial position x and y:", self.get_position())
        if DEBUG: print(self.map)
        if DEBUG: print("solving...")
        best = self.genetic()
        self.current_plan = None
        if DEBUG: print("best:")
        if DEBUG: print(best)
        planning_succeeded = best[1] >= INF / 2
        self.actions = list(best[0]) if planning_succeeded else []
        self.metrics["planning_succeeded"] = planning_succeeded
        self.metrics["nodes_expanded"] = None
        if DEBUG: print("list of actions=" + str(self.actions))
        if not planning_succeeded:
            if DEBUG: print("could not solve")

    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        if len(self.actions) > 0:
            action = self.actions.pop(0)
            return cast(Literal[0, 1, 2, 3], action)
        raise EOFError("No actions!")

    population_size = 20 #TODO
    max_gens = 40 #TODO
    possible_actions = [0, 1, 2, 3]

    # individual = tuple(action1, action2, action3, ... , actionN)
    def generate_random_individual(self):
        return tuple(self.random.choices(self.possible_actions, k=self.max_actions_per_individual))

    def select_parents(self, population, k):
        # another (better) approach: use fitness as a weight
        #for now, return the k with more fitness
        return self.get_best(population, k)
    
    def get_best(self, population, k):
        def get_fitness(item):
            # item = (individual, fitness)
            return item[1]
        
        sorted_pop = sorted(population.items(), key=get_fitness, reverse=True)
        return sorted_pop[:k]

    # receives [(tuple(action1, actionN), fitness), ((tuple(action1, actionN), fitness)]
    # should return just the tuple of actions (not with the fitness)
    def crossover(self, parents):
        result = []
        len_parents = len(parents) - 1
        for i in range(self.max_actions_per_individual):
            parent_idx = self.random.randint(0, len_parents)
            result.append(parents[parent_idx][0][i])
        return tuple(result)

    # receives tuple(action1, actionN)
    # should return just the tuple of actions (not with the fitness)
    def mutate(self, actions):
        rate = 0.1 #TODO
        new_actions = list(actions)

        for i in range(len(new_actions)):
            if self.random.random() < rate:
                new_actions[i] = self.random.choice(self.possible_actions)
        return tuple(new_actions)

    # crossover:
    #   for now, will pick randomly one action from one parent or the other
    #   later, should do based on the path - and join the paths from parent 1 with parent 2 where possible.
    # then, add a few mutation
    # done, generated one child
    # then, select parents again, do crossover and mutation again and repeat untill population is full
    def get_next_gen(self, population):
        new_pop = dict()
        counter = 0
        limit = 10*self.population_size

        # keep best (elitism)
        # for later, increase to K-best?
        best = self.get_best(population, 1)[0]
        if DEBUG: print("best" + str(best))
        new_pop[best[0]]=best[1]

        while len(new_pop) < self.population_size and counter < limit:
            counter += 1
            parents = self.select_parents(population, 2)
            child_actions = self.crossover(parents)
            child_actions = self.mutate(child_actions)

            if child_actions not in new_pop:
                new_pop[child_actions] = 0

        if len(new_pop) < self.population_size:
            if DEBUG: print("coul not fully generate next gen. too much duplicated children")

        return new_pop
 
# TODO: we are pre-computing the path and then running it in gymnasium, so all random seeds need to be set.
# need to make this logic run while the smulation is actually running. 
# this makes it harder because the agent only knows its position and if it is still alive
    def genetic(self):
        population = dict() # population[individual] = fitness
        #generate first individuals
        for i in range(self.population_size):
            individual = self.generate_random_individual()
            population[individual] = 0 # init fitness = 0

        for gen_number in range(self.max_gens):
            if DEBUG: print(gen_number)
            for individual in population:
                self.gym_env.reset(seed=0) # for now, keep the same seed so we can run the GA in the same map and the actions have the same result
                self.metrics["planning_simulation_count"] += 1

                # other options/ideas for fitness
                # fitness1 = INF - steps    if reached goal 
                #            steps          otherwise

                # fitness2 = INF - steps                   if reached goal 
                #           max distance from start     otherwise

                # fitness3: reward unique positions visited (with a weight)
                steps = 0
                reward = 0
                for action in individual:
                    observation, reward, terminated, truncated, info = self.gym_env.step(action)
                    if float(reward) > 0 or float(reward) < 0 or terminated or truncated:
                        break
                    steps += 1

                if float(reward) > 0:
                    new_fitness = INF - steps
                else:
                    new_fitness = steps
                population[individual] = new_fitness

            # print population
            for individual,fitness in population.items():
                if DEBUG: print(f"{fitness}: {individual}")

            # select parents
            parents = self.select_parents(population, 2)
            if DEBUG: print("selected parents:")
            for parent in parents:
                if DEBUG: print("\t"+str(parent))

            next_gen = self.get_next_gen(population)
            if DEBUG: print("next gen:")
            for c in next_gen:
                if DEBUG: print("\t"+str(c))

            #last gen
            if gen_number == self.max_gens -1:
                break
            
            population = next_gen

        return self.get_best(population, 1)[0]
