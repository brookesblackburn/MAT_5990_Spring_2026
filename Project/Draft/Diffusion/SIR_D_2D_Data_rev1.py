# -*- coding: utf-8 -*-
"""
Brookes Heil Blackburn
MAT 5990 Spring 2026 - Final Project Draft
Generate Data 2D with Diffusion
"""
import os
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

# %% Variables
D = 1.5  # Thermal Conductivity/Diffusion coefficient
a = .3  # "burn rate" of those exposed
r = .15  # "Fatality" "Burnt up" Rate

# Total Spatial length ; Number spatial steps ; h
L = 50
J = 150
h = L/J

# Total Time ; Number time steps ; k
TT = 70
k = (h**2)/(D*2) * .5
N = int(TT/k)
# k = TT/N

# Set up x, t and mesh  grid
x = np.linspace(0, L, J)
y = np.linspace(0, L, J)
t = np.linspace(0, TT, N)
[X, Y] = np.meshgrid(x, y)

increment = 10
q_time_print = []
# %% Set up vectors for S, I, R

# "Recovered" matrix initialized

R_temp = np.zeros((J, J, N))
R = np.zeros((J, J, N//increment))
# Susceptibles matrix initialized
S_temp = np.zeros((J, J, N))
S = np.zeros((J, J, N//increment))

# Infected matrix initialized
I_temp = np.zeros((J, J, N))
I = np.zeros((J, J, N//increment))

# Initial Condition
center = L/2
sigma = 10
amplitude = 0.2
I_temp[:, :, 0] = amplitude * \
    np.exp(- (((X - center)**2 + (Y - center)**2) / sigma**2))
# I[:, 0] = .01
S_temp[:, :, 0] = 1 - I_temp[:, :, 0]
R_temp[:, :, 0] = 0.0


# %% Partial Difference Equation (Numerical Scheme)


for n in range(0, N-1):
    # for j in range(1, J-1):  # "x direction" ORIGINALLY WROTE LOOP AI HELPED MAKE MORE EFFICIENT WITH VECTORIZATION
    #     for m in range(1, J-1):  # "y direction"

    S_x = (S_temp[2:, 1:-1, n] - S_temp[:-2, 1:-1, n]) / (2*h)
    S_xx = (S_temp[2:, 1:-1, n] - 2*S_temp[1:-1, 1:-1, n] +
            S_temp[:-2, 1:-1, n]) / (h**2)

    S_y = (S_temp[1:-1, 2:, n] - S_temp[1:-1, :-2, n]) / (2*h)
    S_yy = (S_temp[1:-1, 2:, n] - 2*S_temp[1:-1, 1:-1, n] +
            S_temp[1:-1, :-2, n]) / (h**2)

    I_x = (I_temp[2:, 1:-1, n] - I_temp[:-2, 1:-1, n]) / (2*h)
    I_xx = (I_temp[2:, 1:-1, n] - 2*I_temp[1:-1, 1:-1, n] +
            I_temp[:-2, 1:-1, n]) / (h**2)

    I_y = (I_temp[1:-1, 2:, n] - I_temp[1:-1, :-2, n]) / (2*h)
    I_yy = (I_temp[1:-1, 2:, n] - 2*I_temp[1:-1, 1:-1, n] +
            I_temp[1:-1, :-2, n]) / (h**2)

    R_x = (R_temp[2:, 1:-1, n] - R_temp[:-2, 1:-1, n]) / (2*h)
    R_xx = (R_temp[2:, 1:-1, n] - 2*R_temp[1:-1, 1:-1, n] +
            R_temp[:-2, 1:-1, n]) / (h**2)

    R_y = (R_temp[1:-1, 2:, n] - R_temp[1:-1, :-2, n]) / (2*h)
    R_yy = (R_temp[1:-1, 2:, n] - 2*R_temp[1:-1, 1:-1, n] +
            R_temp[1:-1, :-2, n]) / (h**2)

    S_t = (-a * S_temp[1:-1, 1:-1, n] * I_temp[1:-1, 1:-1, n])
    I_t = ((a * S_temp[1:-1, 1:-1, n] * I_temp[1:-1, 1:-1, n]) -
           (r * I_temp[1:-1, 1:-1, n]) +
           (D * (I_xx + I_yy)))
    R_t = r * I_temp[1:-1, 1:-1, n]

    S_temp[1:-1, 1:-1, n+1] = S_temp[1:-1, 1:-1, n] + (k * S_t)
    I_temp[1:-1, 1:-1, n+1] = I_temp[1:-1, 1:-1, n] + (k * I_t)
    R_temp[1:-1, 1:-1, n+1] = R_temp[1:-1, 1:-1, n] + (k * R_t)

    # Neumann boundary (how would this work with fire growing???..it doesn't it enforces a boundary which causes weird behavior at the corners later in time)

    S_temp[0, :, n+1] = S_temp[1, :, n+1]
    S_temp[J-1, :, n+1] = S_temp[J-2, :, n+1]
    S_temp[:, 0, n+1] = S_temp[:, 1, n+1]
    S_temp[:, J-1, n+1] = S_temp[:, J-2, n+1]

    I_temp[0, :, n+1] = I_temp[1, :, n+1]
    I_temp[J-1, :, n+1] = I_temp[J-2, :, n+1]
    I_temp[:, 0, n+1] = I_temp[:, 1, n+1]
    I_temp[:, J-1, n+1] = I_temp[:, J-2, n+1]

    R_temp[0, :, n+1] = R_temp[1, :, n+1]
    R_temp[J-1, :, n+1] = R_temp[J-2, :, n+1]
    R_temp[:, 0, n+1] = R_temp[:, 1, n+1]
    R_temp[:, J-1, n+1] = R_temp[:, J-2, n+1]

    if n % increment == 0:
        q = int(n / increment)
        S[:, :, q] = S_temp[:, :, n]
        I[:, :, q] = I_temp[:, :, n]
        R[:, :, q] = R_temp[:, :, n]
        q_time_print.append(t[n])

# %% Add up S+I+1 = 1
total_pop = S+I+R
total_pop_mean = total_pop.mean()
print(total_pop_mean)
# %% Save Data Files - S,I,R,S_t,I_t,R_t,I_xx,x,t for use in Manual fit / SINDy and plotting

os.makedirs('data', exist_ok=True)

np.save('data/S.npy', S)
np.save('data/I.npy', I)
np.save('data/R.npy', R)
np.save('data/x.npy', x)
np.save('data/y.npy', y)
np.save('data/t.npy', t[::increment])


# %% Visualize in 3D Contour

time_stamps_early = [10, 40, 100]

for ts in time_stamps_early:
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(x, y, R[:, :, ts], levels=10, cmap='hot_r')
    cs = plt.contour(x, y, R[:, :, ts], levels=5,
                     colors='black', linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
    plt.colorbar(cp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f't = {q_time_print[ts]:.2f}')
    plt.show()

# %% Image show

for ts in time_stamps_early:
    plt.imshow(R[:, :, ts], origin='lower', extent=[0, L, 0, L], cmap='hot_r')
    cs = plt.contour(x, y, R[:, :, ts], levels=5,
                     colors='black', linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
    plt.colorbar(cp)
    plt.title(f't = {q_time_print[ts]:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# %% Plotting
fig1, ax1 = plt.subplots()

colors_early = ['green', 'orange', 'sienna']
midpoint_index = J//2
for i in range(len(time_stamps_early)):
    ts = time_stamps_early[i]
    ax1.plot(x, S[:, midpoint_index, ts], color=colors_early[0], linewidth=1)
    ax1.plot(x, I[:, midpoint_index, ts], color=colors_early[1], linestyle='--',
             linewidth=1)
    ax1.plot(x, R[:, midpoint_index, ts], color=colors_early[2], linewidth=1)

    ax1.set_xlim(left=0, right=L)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")

custom_lines = [
    Line2D([0], [0], color='green',  linewidth=1),
    Line2D([0], [0], color='orange', linewidth=1, linestyle='--'),
    Line2D([0], [0], color='sienna', linewidth=1),
]

custom_labels = [
    f't={q_time_print[time_stamps_early[0]]:.2f},{q_time_print[time_stamps_early[1]]:.2f},{
        q_time_print[time_stamps_early[2]]:.2f}:  Unburned (S)',
    f't={q_time_print[time_stamps_early[0]]:.2f},{q_time_print[time_stamps_early[1]]:.2f},{
        q_time_print[time_stamps_early[2]]:.2f}:  Active Burn (I)',
    f't={q_time_print[time_stamps_early[0]]:.2f},{q_time_print[time_stamps_early[1]]:.2f},{
        q_time_print[time_stamps_early[2]]:.2f}:  Burnt (R)',
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
fig1.text(0.5, 0.91, f'Gaussian IC: centered at x={center:.0f}, y={y[midpoint_index]:.0f}, $\\sigma={
          sigma}$, amplitude={amplitude}', ha='center', fontsize=9, color='grey')


# fig2, ax2 = plt.subplots()

# time_stamps_late = [int(k*N*0.5), int(k*N*0.7), int(k*N*0.9)]
# colors_other = ['lightgrey', 'grey', 'black']
# for i in range(len(time_stamps_late)):
#     ts = time_stamps_late[i]
#     ax2.plot(x, S[:, ts], color=colors_other[2], linewidth=1)
#     ax2.plot(x, I[:, ts], color=colors_other[1], linestyle='--',
#              linewidth=1)
#     ax2.plot(x, R[:, ts], color=colors_other[0], linewidth=2)

# ax2.set_xlim(left=0, right=L)
# ax2.set_xlabel("x")
# ax2.set_ylabel("u")

# # --- Did not like the Legend I was getting thought non-material to ask AI
# # to help with legend
# custom_lines = [
#     Line2D([0], [0], color='black', linewidth=1),
#     Line2D([0], [0], color='grey',   linewidth=1, linestyle='--'),
#     Line2D([0], [0], color='lightgrey',   linewidth=1),
# ]

# custom_labels = [
#     f't={int(k*N*0.5)},{int(k*N*0.7)},{int(k*N*0.90)}: Unburned (S)',
#     f't={int(k*N*0.5)},{int(k*N*0.7)},{int(k*N*0.90)}: Active Burn (I)',
#     f't={int(k*N*0.5)},{int(k*N*0.7)},{int(k*N*0.90)}: Burnt (R)',
# ]

# ax2.legend(custom_lines, custom_labels,
#            loc='upper right',
#            bbox_to_anchor=(1, .90),
#            fontsize=7,
#            framealpha=0.9)
# plt.tight_layout()
# fig2.suptitle(
#     'Proportion Burned Later-On', fontsize=12)
# plt.subplots_adjust(top=0.88)
# fig2.text(0.5, 0.91, f'Gaussian IC: centered at x={
#           center:.0f}, $\\sigma={sigma}$, amplitude={amplitude}', ha='center', fontsize=9, color='grey')

# # fig2.text(0.5, 0.91, f'Uniform IC: $S_0=${S[0, 0]:.2f}, $I_0=${I[0, 0]:.2f}, $R_0=${R[0, 0]:.2f}', ha='center', fontsize=9, color='grey')

# fig3, ax3 = plt.subplots()
# ax3.plot(t, S[10, :], label='S')
# ax3.plot(t, I[10, :])
# ax3.plot(t, R[10, :])
# # ax3.set_xlim(left=400, right=1800)
# ax3.set_xlabel("t")
# ax3.set_ylabel("state")
# plt.show()
