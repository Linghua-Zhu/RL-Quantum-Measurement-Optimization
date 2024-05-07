import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.utils import QuantumInstance

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import math

# See DOI: 10.1103/PhysRevX.6.031007
# Here, we use parameters given for H2 at R=1.75A
g0 = -0.5597
g1 = +0.1615
g2 = -0.0166
g3 = +0.4148
g4 = +0.1226
g5 = +0.1226

nuclear_repulsion = 0.3023869942
Energy_FCI = -0.97516853

def get_probability_distribution_2(counts, NUM_SHOTS):
    #   output_distr = [v/NUM_SHOTS for v in counts.values()]
    for k in {'00', '01', '10', '11'}:
        if k not in counts.keys():
            counts[k] = 0
    sorted_counts = sorted(counts.items())
    # print(sorted_counts)
    output_distr = [v[1] / NUM_SHOTS for v in sorted_counts]
    if len(output_distr) == 1:
        output_distr.append(1 - output_distr[0])
    return output_distr


def simulator(theta, shots):
    E = g0 + nuclear_repulsion

    # Get the expectation value of the first three terms <Z0>, <Z1>, <Z0Z1>
    circ = QuantumCircuit(2, 2)
    circ.name = "H2 STO-3G g1-g3"
    circ.x(1)
    circ.ry(np.pi / 2, 0)
    circ.rx(np.pi / 2, 1)
    circ.cx(0, 1)
    circ.rz(theta[0], 1)
    circ.cx(0, 1)
    circ.ry(theta[1], 0)
    circ.rx(theta[1], 1)
    circ.measure([0, 1], [0, 1])

    # Get the expectation value of <Y0Y1>
    circ_2 = QuantumCircuit(2, 2)
    circ_2.name = "H2 STO-3G g4"
    circ_2.x(1)
    circ_2.ry(np.pi / 2, 0)
    circ_2.rx(np.pi / 2, 1)
    circ_2.cx(0, 1)
    circ_2.rz(theta[0], 1)
    circ_2.cx(0, 1)
    circ_2.ry(theta[1], 0)
    circ_2.rx(theta[1], 1)
    circ_2.sdg(0)
    circ_2.h(0)
    circ_2.sdg(1)
    circ_2.h(1)
    circ_2.measure([0, 1], [0, 1])

    # Get the expectation value of <X0X1>
    circ_3 = QuantumCircuit(2, 2)
    circ_3.name = "H2 STO-3G g5"
    circ_3.x(1)
    circ_3.ry(np.pi / 2, 0)
    circ_3.rx(np.pi / 2, 1)
    circ_3.cx(0, 1)
    circ_3.rz(theta[0], 1)
    circ_3.cx(0, 1)
    circ_3.ry(theta[1], 0)
    circ_3.rx(theta[1], 1)
    circ_3.h(0)
    circ_3.h(1)
    circ_3.measure([0, 1], [0, 1])

    shots1, shots2, shots3 = shots

    # job = qpu_backend.run(circ, shots=shots1)
    job = execute(circ, Aer.get_backend('qasm_simulator'), shots=shots1)
    result = job.result()
    counts = result.get_counts(circ)
    output_distr = get_probability_distribution_2(counts, shots1)

    E1 = -g1 * (output_distr[0] + output_distr[1] - output_distr[2] - output_distr[3])
    E2 = -g2 * (output_distr[0] - output_distr[1] + output_distr[2] - output_distr[3])
    E3 = g3 * (output_distr[0] - output_distr[1] - output_distr[2] + output_distr[3])
    E += E1 + E2 + E3

    # job = qpu_backend.run(circ_2, shots=shots2)
    job = execute(circ_2, Aer.get_backend('qasm_simulator'), shots=shots2)
    result = job.result()
    counts = result.get_counts(circ_2)
    output_distr = get_probability_distribution_2(counts, shots2)
    E += g4 * (output_distr[0] - output_distr[1] - output_distr[2] + output_distr[3])

    # job = qpu_backend.run(circ_3, shots=shots3)
    job = execute(circ_3, Aer.get_backend('qasm_simulator'), shots=shots3)
    result = job.result()
    counts = result.get_counts(circ_3)
    output_distr = get_probability_distribution_2(counts, shots3)
    E += g5 * (output_distr[0] - output_distr[1] - output_distr[2] + output_distr[3])

    return E


class Environ:
    def __init__(self):

        self.observation_space_dim = 1
        self.action_space_dim = 1
        self.std_threshold = 1e-3
        self.max_shots = 1000
        self.initialshots = 9

        self.high = 1
        self.low = 0
        
        self.iter_per_adamcall = 10
        
    def reset(self):

        self.theta = np.array([3.9, 2.0])

        # Initialize the ADAM variables
        self.t = 0
        self.m_t = 0
        self.v_t = 0
        
        self.totalshots = self.initialshots * self.iter_per_adamcall
        
        energies, gradents = self.adam(self.theta, shots=self.initialshots, iters=self.iter_per_adamcall)
        X = np.arange(self.iter_per_adamcall).reshape(-1,1)
        Y = np.array(energies).reshape(-1,1)

        reg = LinearRegression().fit(X, Y)
        R2 = reg.score(X, Y)
        self.min_value = np.mean(energies)

        Y_pred = reg.predict(X)
        mae = mean_absolute_error(Y_pred, Y)
        self.state = np.array([-math.log10(10e-5 if np.abs(reg.coef_[0,0])==0 else np.abs(reg.coef_[0,0])), 0.0])
        
        return self.state

    def step(self, action):
        shots = int(self.max_shots * action)

        if shots<=3:
            shots = 3

        energies, gradents = self.adam(self.theta, shots=shots, iters=self.iter_per_adamcall)
        
        X = np.arange(self.iter_per_adamcall).reshape(-1,1)
        Y = np.array(energies).reshape(-1,1)

        reg = LinearRegression().fit(X, Y)

        Y_pred = reg.predict(X)
        mae = mean_absolute_error(Y_pred, Y)
        R2 = reg.score(X, Y)

        if np.mean(energies)<self.min_value + self.min_value*0.05 or np.mean(energies)<self.min_value - self.min_value*0.05:
            state2 = 1.0
            if np.mean(energies)<self.min_value:
                self.min_value = np.mean(energies)
        else:
            state2 = 0.0
        self.state = np.array([-math.log10(10e-5 if np.abs(reg.coef_[0,0])<10e-5 else np.abs(reg.coef_[0,0])), state2])
        
        ## termination and rewards
        if self.state[0]>3.5 and state2==1:
            done = True
            reward = np.array([20])
        else:
            done = False
            reward = -np.array([shots/self.max_shots])
        print('energy mean: {:.4f}  state1: {:.4f} state2: {:.4f} mae: {:.4f} shots: {}  done: {} totalshots: {}'.format(np.mean(energies), self.state[0], self.state[1], R2, shots, done, self.totalshots))

        return self.state, reward[0], done, None

    def adam(self, theta, shots, iters):
        # initialize something
        base_lr = 0.05
        eps = 0.02
        nit = 0
        x_opt = np.copy(theta)

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        energies = []
        gradents = []
        per_shots = shots // 3
        self.totalshots += per_shots*3*self.iter_per_adamcall
        
        while nit < iters:
            # Compute the function value and gradient
            fval, grad = self.func_and_gradient(x_opt, lambda x: simulator(x, shots=(per_shots, per_shots, per_shots)), eps)
            energies.append(fval)
            gradents.append(np.linalg.norm(grad))
            
            # gradient descent
            x_opt = x_opt - base_lr*grad

            nit += 1

        self.theta = x_opt

        return energies, gradents

    def func_and_gradient(self, x_opt, fun, eps):
        f = fun(x_opt)
        grad = np.zeros_like(x_opt)

        for i in range(x_opt.size):
            x_plus_h = x_opt.copy()
            x_plus_h[i] += eps
            f_plus_h = fun(x_plus_h)
            grad[i] = (f_plus_h - f) / eps

        return f, grad
