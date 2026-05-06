# Brookes Heil Blackburn MAT5990 Spring 2026
# S - I - R --> NO diffusion

import scipy as sp
import numpy as np
import pandas as pd
from sklearn import linear_model
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
# %% variables
beta = .5
gamma = .15
S0 = 0.99
I0 = 0.01
R0 = 0
# %% phi build function


def phi(x, y, z):
    return x, y, z, x*y, y*z, x*z
# %% rhs


def rhs(t, y, beta, gamma):
    return np.array([-beta * (y[0] * y[1]),
                     (beta * (y[0] * y[1]) + (-gamma * y[1])),
                     (gamma * y[1])])


# %%
t = np.linspace(0, 40, 1000)
sol = sp.integrate.solve_ivp(
    rhs, t[[0, -1]], [S0, I0, R0], t_eval=t, args=(beta, gamma))
h = sol.t[1] - sol.t[0]

Phi = np.column_stack(phi(sol.y[0], sol.y[1], sol.y[2]))
Phi = Phi[1:-1]
Y = sol.y.T  # pretend this was my collected data
dydt = (Y[2:] - Y[:-2])/(2*h)  # this is my calculated finite diff dydt
# %% Try to discover the algorithm
alpha_sweep = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for alpha in (alpha_sweep):
    la_S = linear_model.Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    la_I = linear_model.Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    la_R = linear_model.Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)

    # Fit for this loops alpha
    S_fit = la_S.fit(Phi, dydt[:, 0])
    I_fit = la_I.fit(Phi, dydt[:, 1])
    R_fit = la_R.fit(Phi, dydt[:, 2])

    # Theta for this loops alpha
    feature_names = ['S', 'I', 'R', 'SI', 'IR', 'SR']
    theta_S = S_fit.coef_
    theta_I = I_fit.coef_
    theta_R = R_fit.coef_
    theta_summary = np.column_stack([feature_names, theta_S, theta_I, theta_R])

    # Threshold sweep for this alpha
    # threshold_sweep = [0.01, 0.02, 0.03, 0.04, 0.05]
    threshold_sweep = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    for threshold in (threshold_sweep):
        mask_S = (abs(theta_S) > threshold)
        mask_I = (abs(theta_I) > threshold)
        mask_R = (abs(theta_R) > threshold)

        # returns, (x, residuals, rank, s)
        sol_theta_S2 = lstsq(Phi[:, mask_S], dydt[:, 0], rcond=None)
        theta_S2 = sol_theta_S2[0]
        theta_S = np.zeros(6)
        theta_S[mask_S] = theta_S2
        sol_theta_I2 = lstsq(Phi[:, mask_I], dydt[:, 1], rcond=None)
        theta_I2 = sol_theta_I2[0]
        theta_I = np.zeros(6)
        theta_I[mask_I] = theta_I2
        sol_theta_R2 = lstsq(Phi[:, mask_R], dydt[:, 2], rcond=None)
        theta_R2 = sol_theta_R2[0]
        theta_R = np.zeros(6)
        theta_R[mask_R] = theta_R2
        theta_summary_2 = np.column_stack([feature_names, np.round(
            theta_S, 4), np.round(theta_I, 4), np.round(theta_R, 4)])
        theta_cleaned_df = pd.DataFrame(theta_summary_2)
        theta_cleaned_df.insert(1, "Threshold", threshold)
        theta_cleaned_df.insert(1, "Alpha", alpha)
        theta_columns = ("Basis Function", "Alpha", "Threshold", "Theta S",
                         "Theta I", "Theta R")
        theta_cleaned_df.columns = theta_columns
        theta_cleaned_df["Alpha"] = theta_cleaned_df["Alpha"].apply(lambda x:
                                                                    f"{x:.0e}")
        print(f'alpha: {alpha}, threshold: {threshold}, \n {theta_cleaned_df}')
# %% visualize
fig, ax = plt.subplots()
ax.plot(sol.t, sol.y[0], linewidth=1, label='S: Unburnt Proportion')
ax.plot(sol.t, sol.y[1], linewidth=1,
        linestyle='--', label='I: Burning Proportion')
ax.plot(sol.t, sol.y[2], linewidth=1, label='R: Burnt-up Proportion')
ax.legend()
ax.grid()
ax.set_xlabel('Time')
ax.set_ylabel('Proportion in State')

fig.show()


# %% unused for this round
# import os
# from matplotlib.lines import Line2D
# from scipy import integrate
# from sklearn import linear_model
# import seaborn as sns
# import pandas as pd
# from scipy.sparse import diags
# from scipy.integrate import simpson
# from scipy.integrate import trapezoid
