import random
from tqdm import tqdm
from Solver import Solver
from typing import Literal, cast
import numpy as np
import neural_controller

DEBUG = False

INF = 10000000
GOAL = b'G'
HOLE = b'H'
OUTPUT_SIZE = 4

class GeneticSolverNeural(Solver):
    
    map: np.ndarray
    map_height: int
    map_width: int

    def get_position(self):
        return self.gym_env.unwrapped.s

    def __init__(self, gym_env, number_of_hidden_units, population_size, max_gens,
                 mutation_rate,mutation_strength,random_gen_deviation,
                 early_return):
        super().__init__(gym_env)
        self.map = self.gym_env.unwrapped.desc
        self.map_height, self.map_width = self.map.shape
        
        self.random = random.Random(x=self.env_config["seed"])

        # will use one hot encoding
        self.input_size = self.get_input_size()
        self.number_of_hidden_units = number_of_hidden_units
        self.population_size = population_size
        self.max_gens = max_gens
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.random_gen_deviation = random_gen_deviation
        self.early_return = early_return
        
        self.rng = np.random.default_rng(seed=self.env_config["seed"])
        self.model_size = (self.input_size * self.number_of_hidden_units) + (self.number_of_hidden_units * OUTPUT_SIZE) + (self.number_of_hidden_units + OUTPUT_SIZE)
        self.model = np.empty(self.model_size, dtype=float)

        if DEBUG: print(f"{self.__class__.__name__} initialized with map:")
        if DEBUG: print("map height:", self.map_height)
        if DEBUG: print("map width:", self.map_width)
        if DEBUG: print("initial position x and y:", self.get_position())
        if DEBUG: print(self.map)
        if DEBUG: print("solving...")
        
        best_model, best_fitness = self.genetic()

        self.model = list(best_model)
        self.metrics["planning_succeeded"] = best_fitness > 0
        self.metrics["nodes_expanded"] = None

    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        nn_input = self.get_nn_input(self.get_position())
        action = neural_controller.get_next_action(
                    self.model,
                    nn_input,
                    self.input_size,
                    self.number_of_hidden_units,
                    OUTPUT_SIZE
                )

        return cast(Literal[0, 1, 2, 3], action)

    def get_input_size(self):
        return self.map_height * self.map_width

    def get_nn_input(self, position):
        nn_input = np.zeros(self.input_size)
        nn_input[position] = 1
        return nn_input

    # individual = model = [w1,w2,...]
    def generate_random_individual(self):
        return tuple(self.rng.normal(0, self.random_gen_deviation, self.model_size)) #center, deviation

    def select_parents(self, population, k):
        # another (better) approach: use fitness as a weight
        #for now, return the k with more fitness
        return self.get_best(population, k)
    
    def get_best(self, population, k):
        #print("getting best...")
        #print(population)
        def get_fitness(item):
            # item = (individual, fitness)
            return item[1]
        
        sorted_pop = sorted(population.items(), key=get_fitness, reverse=True)
        return sorted_pop[:k]

    # receives models with their fitness, returns one model (only model, without fitness)
    def crossover(self, parents):
        #print("crossing over...")
        #print(parents)
        result = []
        len_parents = len(parents) - 1
        for i in range(self.model_size):
            parent_idx = self.random.randint(0, len_parents)
            result.append(parents[parent_idx][0][i])
        #print(result)
        #exit()
        return tuple(result)

    # receives model, returns model with mutated values
    def mutate(self, model):
        #print("mutating...")
        model = list(model)
        #print(model)
        for i in range(len(model)):
            if self.random.random() < self.mutation_rate:
                model[i] += self.rng.normal(0, self.mutation_strength)
        #print(model)
        return tuple(model)

    # crossover:
    #   pick randomly one entry from one parent or the other
    # then, mutation
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
            children = self.crossover(parents)
            children = self.mutate(children)

            if children not in new_pop:
                new_pop[children] = 0

        if len(new_pop) < self.population_size:
            print("coul not fully generate next gen. too much duplicated children")

        return new_pop

    def evaluate_individual(self, model):
        position, _ = self.gym_env.reset(seed=self.env_config["seed"]) # for now, keep the same seed so we can run the GA in the same map and the actions have the same result
        last_pos = position
        self.metrics["planning_simulation_count"] += 1

        score = 0.0
        goal_reached = False

        visited = set()
        max_steps = self.env_config["max_episode_steps"]
        for _ in range(max_steps):

            nn_input = self.get_nn_input(position)
            action = neural_controller.get_next_action(
                        model,
                        nn_input,
                        self.input_size,
                        self.number_of_hidden_units,
                        OUTPUT_SIZE
                    )
            
            position, reward, terminated, truncated, info = self.gym_env.step(action)
            #print(f"State: {position}, Action: {action}")
            #print(f"Score: {score}, Unique states: {len(visited)}")

            score -= 0.1

            if last_pos == position: #did not move
                score -= 1.0
            
            last_pos = position
            
            if position in visited:
                score -= 0.7
            else:
                score += 0.2
            visited.add(position)

            if float(reward) > 0: #goal
                score += 1000
                goal_reached = True
                break
            if float(reward) < 0: #hole
                score -= 1000
                break
            
            if terminated or truncated:
                #print("end!")
                break
        #print("\t"+str(score))
        return score, goal_reached

    # Note: we are pre-computing the best model and then running it in gymnasium, so all random seeds need to be set.
    def genetic(self):
        population = dict() # population[individual] = fitness
        #generate first individuals
        for i in range(self.population_size):
            individual = self.generate_random_individual()
            # print(individual)
            population[individual] = 0 # init fitness = 0

        with tqdm(total=self.max_gens * len(population), desc="Progress") as progress_bar:
            for gen_number in range(self.max_gens):
                if DEBUG: print(gen_number)
                for individual in population:
                    progress_bar.update(1)
                    new_fitness, goal_reached = self.evaluate_individual(individual)
                    population[individual] = new_fitness
                    #print(f"Generation {gen_number}: fitness = {new_fitness}")
                    if self.early_return and goal_reached:
                        return self.get_best(population, 1)[0]
                
                # print population
                if DEBUG:
                    for individual,fitness in population.items():
                        print(f"{fitness}: {individual}")

                # select parents
                parents = self.select_parents(population, 2)
                if DEBUG: 
                    print("selected parents:")
                    for parent in parents:
                        print("\t"+str(parent))

                next_gen = self.get_next_gen(population)

                #best = self.get_best(population, 1)[0]
                #print(f"Generation {gen_number}: best fitness = {best[1]}")

                #last gen
                if gen_number == self.max_gens -1:
                    break
                
                population = next_gen

        best = self.get_best(population, 1)[0]
        print(f"best fitness = {best[1]}")
        return best