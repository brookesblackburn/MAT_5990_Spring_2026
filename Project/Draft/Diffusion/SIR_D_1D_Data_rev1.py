# -*- coding: utf-8 -*-
"""
Brookes Heil Blackburn
MAT 5990 Spring 2026 - Final Project Draft
Generate Data for 1D with Diffusion
"""
import os
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

# %% Variables
D = 1.5  # Thermal Conductivity/Diffusion coefficient
a = 1.5  # "burn rate" of those exposed
r = .15  # "Fatality" "Burnt up" Rate

# Total Spatial length ; Number spatial steps ; h
L = 200
J = 400
h = L/J

# Total Time ; Number time steps ; k
TT = 70
k = (h**2)/(D*2)
N = int(TT/k)
# k = TT/N

# Set up x, t and mesh  grid
x = np.linspace(0, L, J)
t = np.linspace(0, TT, N)
[T, X] = np.meshgrid(t, x)

# %% Set up vectors for S, I, R

# "Recovered" matrix initialized - preserves conserved system
R = np.zeros((J, N))
# Susceptibles matrix initialized
S = np.zeros((J, N))
# Infected matrix initialized
I = np.zeros((J, N))

# Initial Condition
center = L/2
sigma = 10
amplitude = 0.2
I[:, 0] = amplitude * np.exp(-((x - center)/sigma)**2)
# I[:, 0] = .01
S[:, 0] = 1 - I[:, 0]
R[:, 0] = 0.0


# %% Set up Vectors for derivatives
# dS/dt = -aSI
# dI/dt = aSI - rI
# dR/dt = rI
I_t = np.zeros((J, N))
S_t = np.zeros((J, N))
R_t = np.zeros((J, N))
S_x = np.zeros((J, N))
S_xx = np.zeros((J, N))
I_x = np.zeros((J, N))
I_xx = np.zeros((J, N))
R_x = np.zeros((J, N))
R_xx = np.zeros((J, N))


# %% Partial Difference Equation (Numerical Scheme)
for n in range(0, N-1):
    for j in range(1, J-1):
        S_x[j, n] = (S[j+1, n] - S[j-1, n]) / (2*h)
        S_xx[j, n] = (S[j+1, n] - 2*S[j, n] + S[j-1, n]) / (h**2)

        I_x[j, n] = (I[j+1, n] - I[j-1, n]) / (2*h)
        I_xx[j, n] = (I[j+1, n] - 2*I[j, n] + I[j-1, n]) / (h**2)

        R_x[j, n] = (R[j+1, n] - R[j-1, n]) / (2*h)
        R_xx[j, n] = (R[j+1, n] - 2*R[j, n] + R[j-1, n]) / (h**2)

        S_t[j, n] = (-a*S[j, n]*I[j, n])
        I_t[j, n] = ((a*S[j, n]*I[j, n]) -
                     (r*I[j, n]) +
                     (D * I_xx[j, n]))
        R_t[j, n] = r*I[j, n]

        S[j, n+1] = S[j, n] + k * S_t[j, n]
        I[j, n+1] = I[j, n] + (k * I_t[j, n])
        R[j, n+1] = R[j, n] + k * R_t[j, n]

    # Neumann boundary (how would this work with fire growing???..it doesn't it enforces a boundary)
    S[0, n+1] = S[1, n+1]
    S[J-1, n+1] = S[J-2, n+1]

    I[0, n+1] = I[1, n+1]
    I[J-1, n+1] = I[J-2, n+1]

    R[0, n+1] = R[1, n+1]
    R[J-1, n+1] = R[J-2, n+1]
# %% Add up S+I+1 = 1
total_pop = S+I+R
total_pop_mean = total_pop.mean()
print(total_pop_mean)


# %% Plotting
fig1, ax1 = plt.subplots()
time_stamps_early = [int(k*N*0.1), int(k*N*0.25), int(k*N*0.3)]

colors_early = ['green', 'orange', 'sienna']

for i in range(len(time_stamps_early)):
    ts = time_stamps_early[i]
    ax1.plot(x, S[:, ts], color=colors_early[0], linewidth=1)
    ax1.plot(x, I[:, ts], color=colors_early[1], linestyle='--',
             linewidth=1)
    ax1.plot(x, R[:, ts], color=colors_early[2], linewidth=1)

    ax1.set_xlim(left=0, right=L)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")

custom_lines = [
    Line2D([0], [0], color='green',  linewidth=1),
    Line2D([0], [0], color='orange', linewidth=1, linestyle='--'),
    Line2D([0], [0], color='sienna', linewidth=1),
]

custom_labels = [
    f't={int(k*N*0.1)},{int(k*N*0.25)},{int(k*N*0.30)}:  Unburned (S)',
    f't={int(k*N*0.1)},{int(k*N*0.25)},{int(k*N*0.30)}:  Active Burn (I)',
    f't={int(k*N*0.1)},{int(k*N*0.25)},{int(k*N*0.30)}:  Burnt (R)',
]

ax1.legend(custom_lines, custom_labels,
           loc='upper right',
           bbox_to_anchor=(1.0, 0.93),
           fontsize=7,
           framealpha=0.9)
plt.tight_layout()
fig1.suptitle(
    'Proportion Burned Early-On', fontsize=12)
plt.subplots_adjust(top=0.88)
fig1.text(0.5, 0.91, f'Gaussian IC: centered at x={
          center:.0f}, $\\sigma={sigma}$, amplitude={amplitude}', ha='center', fontsize=9, color='grey')


fig2, ax2 = plt.subplots()

time_stamps_late = [int(k*N*0.5), int(k*N*0.7), int(k*N*0.9)]
colors_other = ['lightgrey', 'grey', 'black']
for i in range(len(time_stamps_late)):
    ts = time_stamps_late[i]
    ax2.plot(x, S[:, ts], color=colors_other[2], linewidth=1)
    ax2.plot(x, I[:, ts], color=colors_other[1], linestyle='--',
             linewidth=1)
    ax2.plot(x, R[:, ts], color=colors_other[0], linewidth=2)

ax2.set_xlim(left=0, right=L)
ax2.set_xlabel("x")
ax2.set_ylabel("u")

custom_lines = [
    Line2D([0], [0], color='black', linewidth=1),
    Line2D([0], [0], color='grey',   linewidth=1, linestyle='--'),
    Line2D([0], [0], color='lightgrey',   linewidth=1),
]

custom_labels = [
    f't={int(k*N*0.5)},{int(k*N*0.7)},{int(k*N*0.90)}: Unburned (S)',
    f't={int(k*N*0.5)},{int(k*N*0.7)},{int(k*N*0.90)}: Active Burn (I)',
    f't={int(k*N*0.5)},{int(k*N*0.7)},{int(k*N*0.90)}: Burnt (R)',
]

ax2.legend(custom_lines, custom_labels,
           loc='upper right',
           bbox_to_anchor=(1, .90),
           fontsize=7,
           framealpha=0.9)
plt.tight_layout()
fig2.suptitle(
    'Proportion Burned Later-On', fontsize=12)
plt.subplots_adjust(top=0.88)
fig2.text(0.5, 0.91, f'Gaussian IC: centered at x={
          center:.0f}, $\\sigma={sigma}$, amplitude={amplitude}', ha='center', fontsize=9, color='grey')


fig3, ax3 = plt.subplots()
ax3.plot(t, S[10, :], label='S')
ax3.plot(t, I[10, :])
ax3.plot(t, R[10, :])
ax3.set_xlabel("t")
ax3.set_ylabel("state")
plt.show()

# %% notes to self
# custom lines and custom labels help from AI
# Diffusion and infection different from 2D
