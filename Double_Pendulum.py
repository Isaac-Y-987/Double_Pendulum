"""
Borrows heavily from this source: https://scipython.com/blog/the-double-pendulum/
"""

import sys
import numpy as np
from numpy import pi
from scipy.integrate import odeint
import tkinter
from math import cos, sin, sqrt
import random

#
# USER-DEFINED CONSTANTS
#
g = 9.81
L1, L2 = 150, 150
m1, m2 = 1, 1
tmax = 500       # simulation duration; everything is precalculated by scipy.integrate.odeint
dt = 1/3 # timestep duration
seed = 99
random.seed(seed)
theta1_init, theta2_init = pi,pi-1
# ^ random.uniform(0, 2 * pi), random.uniform(0, 2 * pi)   # initial angles
energy_scaling_factor = None    #pixles per joules


#
# HELPER FUNCTIONS
#
def deriv(y: np.array, t: float, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, omega1, theta2, omega2 at a specific point in time t.
    :param y:   1D array containing the following, in order; [theta1, omega1, theta2, omega2]
    :param t:   timestamp in seconds
    :param L1:  length of first arm, meters
    :param L2:  length of second arm, meters
    :param m1:  mass of first bob, kilograms
    :param m2:  mass of second bob, kilograms
    """
    theta1, z1, theta2, z2 = y  # z1 and z2 are just the derivatives of theta1 and theta2 with respect to time

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot


y = [theta1_init, 0, theta2_init, 0]
tlist = np.arange(0, tmax, dt)
result = odeint(deriv, y, tlist, (L1, L2, m1, m2))
pos_1 = result[:,0]
pos_2 = result[:,2]

counter = 0

def simulation_loop():
    """Repeatedly call show_simulation"""
    global counter
    delay = 50   # ms
    theta1 = pos_1[counter]
    theta2 = pos_2[counter]
    omega1 = result[counter, 1]
    omega2 = result[counter, 3]
    show_simulation(pos_1[counter], pos_2[counter], L1, L2, theta1, theta2, omega1, omega2)
    counter += 1
    double_pendulum.after(delay, simulation_loop)


def show_simulation(pos_bob1, pos_bob2, L1, L2, theta1, theta2, omega1, omega2):
    """
    This function clears and updates the canvas
    """

    #Converts pos_bob1 and pos_bob2 into cartesian from polar
    x_pos_bob1 = sin(pos_bob1) * L1
    y_pos_bob1 = -cos(pos_bob1) * L1
    x_pos_bob2 = sin(pos_bob2) * L2 + x_pos_bob1
    y_pos_bob2 = -cos(pos_bob2) * L2 + y_pos_bob1

    x_pos_bob1 += 640
    y_pos_bob1 = 360 - y_pos_bob1
    x_pos_bob2 += 640
    y_pos_bob2 = 360- y_pos_bob2

    PE_1, KE_1, PE_2, KE_2 = get_energy(m1, m2, theta1, theta2, omega1, omega2, L1, L2, g)

    global energy_scaling_factor
    if energy_scaling_factor is None:
        energy_scaling_factor = 680/sum(get_energy(m1, m2, theta1, theta2, omega1, omega2, L1, L2, g))

    tick_mark_PE1 = PE_1 * energy_scaling_factor + 300
    tick_mark_KE_1 = KE_1 * energy_scaling_factor + tick_mark_PE1
    tick_mark_PE_2 = PE_2 * energy_scaling_factor + tick_mark_KE_1
    tick_mark_KE_2 = KE_2 * energy_scaling_factor + tick_mark_PE_2

    global double_pendulum_canvas
    double_pendulum_canvas.delete("all")


    double_pendulum_canvas.create_rectangle(300, 100, 980, 25, fill="black")
    double_pendulum_canvas.create_rectangle(300, 100, tick_mark_PE1, 25, fill = "#2487b5")
    double_pendulum_canvas.create_rectangle(tick_mark_PE1, 100, tick_mark_KE_1, 25, fill="#ed9332")
    double_pendulum_canvas.create_rectangle(tick_mark_KE_1, 100, tick_mark_PE_2, 25, fill = "#2487b5")
    double_pendulum_canvas.create_rectangle(tick_mark_PE_2, 100, tick_mark_KE_2, 25, fill="#ed9332")

    print("KE2 = " + str(KE_2))
    print("KE1 = " + str(KE_1))
    print("total energy = " + str(PE_1 + KE_1 + PE_2 + KE_2))
    double_pendulum_canvas.create_oval(635, 355, 645, 365, fill="white")

    double_pendulum_canvas.create_line(640, 360, x_pos_bob1, y_pos_bob1, fill="white")
    double_pendulum_canvas.create_oval(x_pos_bob1 - 25, y_pos_bob1 - 25, x_pos_bob1 + 25, y_pos_bob1 + 25, fill="white")
    double_pendulum_canvas.create_line(x_pos_bob1, y_pos_bob1, x_pos_bob2, y_pos_bob2, fill="white")
    double_pendulum_canvas.create_oval(x_pos_bob2 - 25, y_pos_bob2 - 25, x_pos_bob2 + 25, y_pos_bob2 + 25, fill="white")



"""
    KE = 0.5 * m * v ^ 2
    PE = m * g * h
    v = angular_velocity * r
    double_pendulum_canvas.update()
"""

def get_energy(m1, m2, theta1, theta2, omega1, omega2, l1, l2, g):
    """[PE_1, KE_1, PE_2, KE_2]"""
    height1 = -cos(theta1) * l1 + l1 #removed +l2 to make PE_1 work
    height2 = -cos(theta2) * l2 + -cos(theta1) * l1 + l1 + l2
    PE_1 = m1 * g * height1
    KE_1 = 0.5 * m1 * (omega1 * l1)**2
    PE_2 = m2 * g * height2
    KE_2 = 0.5 * m2 * (((cos(theta2) * omega2 * l2) + (cos(theta1) * omega1 * l1))**2 + ((sin(theta2) * omega2 * l2) + (sin(theta1) * omega1 * l1))**2)
    sum = PE_1+PE_2+KE_1+KE_2
    return [PE_1, KE_1, PE_2, KE_2]

double_pendulum = tkinter.Tk()
double_pendulum.title("Double Pendulum")
double_pendulum.geometry("1280x720")
double_pendulum_canvas = tkinter.Canvas(double_pendulum, height=720, width=1280, background="#6e6e6e")
    # Creates a new canvas widget and sets some basic parameters.
double_pendulum_canvas.grid(row=0, column=0)
    # "grid" tells a widget *where* on the window it should go. Row + column numbers start at 0
double_pendulum.after(500, simulation_loop)
double_pendulum.mainloop()
