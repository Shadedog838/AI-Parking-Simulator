# Author: Shandon Mith, sam8031

import math
import time
import multiprocessing
import random
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt


def binary_to_gray_op(n, length):
    n = int(n, 2)
    n ^= (n >> 1)

    res = bin(n)[2:]
    return str(res).zfill(length)


def gray_decode(n):
    m = n >> 1
    while m:
        n ^= m
        m >>= 1
    return n


class Individual:

    def __init__(self, name, gammas, betas, code, cost, fitness, fitness_ratio):
        self.id = name
        self.gammas = gammas
        self.betas = betas
        self.code = code
        self.cost = cost
        self.fitness = fitness
        self.fitness_ratio = fitness_ratio
        self.states = {}


class Parking:
    def __init__(self, population_size: int, ga_params: int, bin_length: int, mutation_rate: float, K: int,
                 total_fitness):
        self.population_size = population_size
        self.ga_params = ga_params
        self.bin_length = bin_length
        self.mutation_rate = mutation_rate
        self.K = K
        self.gen = {}
        self.total_fitness = total_fitness
        self.tolerance = 0.1
        self.complete = False

    def random_generation(self):
        choices = ["0", "1"]
        binary_code = ""
        for i in range(self.bin_length * 2):
            binary_code = binary_code + random.choice(choices)
        gray_code1 = binary_to_gray_op(binary_code[:7], self.bin_length)
        gray_code2 = binary_to_gray_op(binary_code[7:], self.bin_length)
        return [gray_code1, gray_code2]

    def get_parameter_values(self, gray_code1, gray_code2) -> list:
        gammas = []
        betas = []
        decimal_gamma = gray_decode(int(gray_code1, 2))
        decimal_beta = gray_decode(int(gray_code2, 2))
        range_heading_angle_rate = 0.524 - (-0.524)
        range_acceleration = 5 - (-5)
        param_value_gamma = (decimal_gamma / (pow(2, self.bin_length) - 1)) * range_heading_angle_rate + -0.524
        param_value_beta = (decimal_beta / (pow(2, self.bin_length) - 1)) * range_acceleration + (-5)
        gammas.append(param_value_gamma)
        betas.append(param_value_beta)
        return [gammas, betas, gray_code1 + gray_code2]

    def get_first_gen(self):
        for i in range(self.population_size):
            gammas = []
            betas = []
            code = ""
            for j in range(self.ga_params):
                gray_codes = self.random_generation()
                gray_code1 = gray_codes[0]
                gray_code2 = gray_codes[1]
                params = self.get_parameter_values(gray_code1, gray_code2)
                gammas = gammas + params[0]
                betas = betas + params[1]
                code = code + params[2]

            individual = Individual(i, gammas, betas, code, 0, 0, 0)
            self.gen[i] = individual

    def get_fitness(self):
        top_individual = {}
        top_fitness = 0
        for individual in self.gen:
            time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            f_gamma = CubicSpline(time, self.gen[individual].gammas, bc_type='natural')
            f_beta = CubicSpline(time, self.gen[individual].betas, bc_type='natural')
            t_new = np.linspace(0, 10, 100)
            g_new = f_gamma(t_new)
            b_new = f_beta(t_new)
            h = 0.1
            t = np.arange(0, 10 + h, h)
            state_final = {
                "x": 0,
                "y": 0,
                "alpha": 0,
                "velocity": 0
            }

            states = {
                0: {
                    "x": 0,
                    "y": 8,
                    "alpha": 0,
                    "velocity": 0,
                    "c": 0
                }
            }
            f_x = lambda velocity, alpha: velocity * math.cos(alpha)
            f_y = lambda velocity, alpha: velocity * math.sin(alpha)
            for i in range(0, len(t) - 1):
                states[i + 1] = {}
                states[i + 1]["x"] = states[i]["x"] + (h * f_x(states[i]["velocity"], states[i]["alpha"]))
                states[i + 1]["y"] = states[i]["y"] + (h * f_y(states[i]["velocity"], states[i]["alpha"]))
                states[i + 1]["alpha"] = states[i]["alpha"] + (g_new[i] * h)
                states[i + 1]["velocity"] = states[i]["velocity"] + (b_new[i] * h)

                if states[i + 1]["x"] <= -4 and states[i + 1]["y"] <= 3:
                    c = math.pow((3 - states[i + 1]["y"]), 2)
                elif (-4 < states[i + 1]["x"] < 4) and states[i + 1]["y"] <= -1:
                    c = math.pow((-1 - states[i + 1]["y"]), 2)
                elif states[i + 1]["x"] >= 4 and states[i + 1]["y"] <= 3:
                    c = math.pow((3 - states[i + 1]["y"]), 2)
                else:
                    c = 0

                states[i + 1]["c"] = states[i]["c"] + c

            if states[len(states) - 1]["c"] == 0:
                J = math.sqrt((pow(states[len(states) - 1]["x"] - state_final["x"], 2)) +
                              (pow(states[len(states) - 1]["y"] - state_final["y"], 2)) +
                              (pow(states[len(states) - 1]["alpha"] - state_final["alpha"], 2)) +
                              (pow(states[len(states) - 1]["velocity"] - state_final["velocity"], 2)))
            else:
                J = self.K + states[len(states) - 1]["c"]

            self.gen[individual].states = states
            self.gen[individual].cost = J
            g = 1 / (J + 1)
            self.gen[individual].fitness = g
            if self.gen[individual].fitness > top_fitness:
                top_individual = individual
                top_fitness = self.gen[individual].fitness
            self.total_fitness = self.total_fitness + g
        return top_individual

    def mutation(self, child_code: str) -> str:
        choices = ["yes", "no"]
        child_code = list(child_code)
        for i in range(0, len(child_code)):
            flip = random.choices(choices, weights=[self.mutation_rate, 1 - self.mutation_rate], k=1)
            if flip[0] == "yes":
                child_code[i] = '0' if child_code[i] == '1' else '1'
        child_code = ''.join(child_code)
        return child_code

    def mate_individuals(self, choices, weights):
        parents = random.choices(choices, weights, k=2)
        parent_one = parents[0]
        parent_two = parents[1]
        crossover_index = random.randint(1, len(parent_one.code) - 1)
        child_one = parent_one.code[:crossover_index]
        chid_two = parent_two.code[:crossover_index]
        child_one = child_one + parent_two.code[crossover_index:]
        chid_two = chid_two + parent_one.code[crossover_index:]
        return [self.mutation(child_one), self.mutation(chid_two)]

    def get_next_generation(self, top_individual):
        choices = []
        weights = []
        new_gen = {}
        self.gen[top_individual].name = 0
        new_gen[self.gen[top_individual].name] = self.gen[top_individual]
        for individual in self.gen:
            self.gen[individual].fitness_ratio = (self.gen[individual].fitness / self.total_fitness)
            choices.append(self.gen[individual])
            weights.append(self.gen[individual].fitness_ratio)

        i = 1
        while len(new_gen) < 200:
            childrens = self.mate_individuals(choices, weights)
            for child in childrens:
                gammas = []
                betas = []
                code = ""
                start = 0
                end = 14
                for j in range(0, self.ga_params):
                    gray_codes = child[start:end]
                    gray_code1 = gray_codes[:7]
                    gray_code2 = gray_codes[7:]
                    params = self.get_parameter_values(gray_code1, gray_code2)
                    gammas = gammas + params[0]
                    betas = betas + params[1]
                    code = code + params[2]
                    start = start + 14
                    end = start + 14
                individual = Individual(i, gammas, betas, code, 0, 0, 0)
                new_gen[i] = individual
                i = i + 1
        self.gen = new_gen

    def plot_solution(self, top_individual):
        x_values = []
        y_values = []
        alpha_values = []
        velocity_values = []
        h = 0.1  # Step size
        t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        time = np.arange(0, 10 + h, h)
        gamma_values = self.gen[top_individual].gammas
        beta_values = self.gen[top_individual].betas
        for state in self.gen[top_individual].states.values():
            x_values.append(state["x"])
            y_values.append(state["y"])
            alpha_values.append(state["alpha"])
            velocity_values.append((state["velocity"]))

        point1 = [-15, 3]
        point2 = [-4, 3]
        point3 = [-4, -1]
        point4 = [4, -1]
        point5 = [4, 3]
        point6 = [15, 3]
        x_obs_points = np.array([point1[0], point2[0], point3[0], point4[0], point5[0], point6[0]])
        y_obs_points = np.array([point1[1], point2[1], point3[1], point4[1], point5[1], point6[1]])
        x_solution = np.array(x_values)
        y_solution = np.array(y_values)
        alpha_points = np.array(alpha_values)
        velocity_points = np.array(velocity_values)
        f_gamma = CubicSpline(t, gamma_values, bc_type='natural')
        f_beta = CubicSpline(t, beta_values, bc_type='natural')
        t_new = np.linspace(0, 10, 100)
        gamma_points = f_gamma(t_new)
        beta_points = f_beta(t_new)

        plt.figure(1)
        plt.ylim(-10.55, 15.99)
        plt.grid()
        plt.title('Solution trajectory')
        plt.xlabel('x (ft)')
        plt.ylabel('y (ft)')
        plt.plot(x_obs_points, y_obs_points, color='black')
        plt.plot(x_solution, y_solution, color='green')

        plt.figure(2)
        plt.title('State histories for x')
        plt.xlabel('Time (s)')
        plt.ylabel('x (ft)')
        plt.grid()
        plt.plot(time, x_solution, color='blue')

        plt.figure(3)
        plt.title('State histories for y')
        plt.xlabel('Time (s)')
        plt.ylabel('y (ft)')
        plt.grid()
        plt.plot(time, y_solution, color='blue')

        plt.figure(4)
        plt.title('State histories for α')
        plt.xlabel('Time (s)')
        plt.ylabel('α (rad)')
        plt.grid()
        plt.plot(time, alpha_points, color='blue')

        plt.figure(5)
        plt.title('State histories for v')
        plt.xlabel('Time (s)')
        plt.ylabel('v (ft/s)')
        plt.grid()
        plt.plot(time, velocity_points, color='blue')

        plt.figure(6)
        plt.title('State histories for γ')
        plt.xlabel('Time (s)')
        plt.ylabel('γ (ft/s\u00b2)')
        plt.grid()
        plt.plot(t_new, gamma_points, color='blue')

        plt.figure(7)
        plt.title('State histories for β')
        plt.xlabel('Time (s)')
        plt.ylabel('β (ft/s\u00b2)')
        plt.grid()
        plt.plot(t_new, beta_points, color='blue')

        plt.show()

    def print_controls(self, top_individual):
        with open("controls.dat", "w", encoding="utf-8") as f:
            for i in range(0, self.ga_params):
                f.write("Y" + str(i) + ": " + str(self.gen[top_individual].gammas[i]) + "\n")
                f.write("β" + str(i) + ": " + str(self.gen[top_individual].betas[i]) + "\n")

    def start(self):
        gen = 0
        self.get_first_gen()
        top_individual = self.get_fitness()
        print("Generation " + str(gen) + " :  J = " + str(self.gen[top_individual].cost))
        gen = gen + 1
        while self.gen[top_individual].cost > self.tolerance:
            if gen == 1200:
                break
            self.get_next_generation(top_individual)
            top_individual = self.get_fitness()
            print("Generation " + str(gen) + " :  J = " + str(self.gen[top_individual].cost))
            gen = gen + 1

        print("\n")
        print("Final state values:")
        print("x_f = " + str(self.gen[top_individual].states[len(self.gen[top_individual].states) - 1]["x"]))
        print("y_f = " + str(self.gen[top_individual].states[len(self.gen[top_individual].states) - 1]["y"]))
        print("alpha_f = " + str(self.gen[top_individual].states[len(self.gen[top_individual].states) - 1]["alpha"]))
        print("v_f = " + str(self.gen[top_individual].states[len(self.gen[top_individual].states) - 1]["velocity"]))

        self.print_controls(top_individual)
        self.plot_solution(top_individual)
        self.complete = True


def main():
    parking = Parking(201, 11, 7, .005, 200, 0)
    p = multiprocessing.Process(target=parking.start(), name="parking")
    p.start()
    sleep_time = 420
    while parking.complete is False and sleep_time != 0:
        sleep_time = sleep_time - 1
        time.sleep(1)
    p.terminate()
    p.join()
    pass


if __name__ == '__main__':
    main()
