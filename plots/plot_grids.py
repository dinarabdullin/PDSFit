''' Plot integration grids '''

import numpy as np

import plots.set_backend
import matplotlib.pyplot as plt
import plots.set_style
from supplement.definitions import const
from mathematics.histogram import histogram

r_increment = 0.01
j_increment = 0.01
ang_increment = 1
threshold = 1e-10


def plot_grid(fig, x, y, axes_labels = ['Parameter (arb. u.)', 'Relative probability (arb. u.)']):
    axes = fig.gca()
    axes.plot(x, y / np.amax(y), 'k-')
    axes.set_xlim(np.amin(x), np.amax(x))
    axes.set_xlabel(axes_labels[0])
    axes.set_ylabel(axes_labels[1])


def plot_grids(r_values, r_weights, j_values, j_weights, xi_values, xi_weights, phi_values, phi_weights, 
               alpha_values, alpha_weights, beta_values, beta_weights, gamma_values, gamma_weights):                   
    fig = plt.figure(facecolor='w', edgecolor='w')
    # P(r)
    if r_values.size == 1 or all(v - r_values[0] < threshold for v in r_values):
        r_axis = np.arange(r_values[0]-1, r_values[0]+1, r_increment)
    else:
        r_axis = np.arange(np.amin(r_values)-1, np.amax(r_values)+1, r_increment)
    if r_weights != []:
        r_prob = histogram(r_values, bins=r_axis, weights=r_weights)
    else:
        r_prob = histogram(r_values, bins=r_axis)
    plt.subplot(2, 4, 1)
    plot_grid(fig, r_axis, r_prob, [r'$\mathit{r}$ (nm)', r'$\mathit{P(r)}$ (arb. u.)'])
    # P(J)
    if j_values.size == 1 or all(v - j_values[0] < threshold for v in j_values):
        j_axis = np.arange(j_values[0]-1, j_values[0]+1, r_increment)
    else:
        j_axis = np.arange(np.amin(j_values)-1, np.amax(j_values)+1, r_increment)
    if j_weights != []:
        j_prob = histogram(j_values, bins=j_axis, weights=j_weights)
    else:
        j_prob = histogram(j_values, bins=j_axis)
    plt.subplot(2, 4, 2)
    plot_grid(fig, j_axis, j_prob, [r'$\mathit{J}$ (MHz)', r'$\mathit{P(J)}$ (arb. u.)'])
    # P(xi) * sin(xi)
    xi_values *= const['rad2deg']
    if xi_values.size == 1 or all(v - xi_values[0] < threshold for v in xi_values):
        xi_axis = np.arange(0, 180, ang_increment)
    else:
        xi_axis = np.arange(np.amin(xi_values)-10, np.amax(xi_values)+10, ang_increment)
    if xi_weights != []:
        xi_prob = histogram(xi_values, bins=xi_axis, weights=xi_weights)
    else:
        xi_prob = histogram(xi_values, bins=xi_axis)
    plt.subplot(2, 4, 3)
    plot_grid(fig, xi_axis, xi_prob, [r'$\mathit{\xi}$ $^\circ$', r'$\mathit{P(\xi)}$ (arb. u.)'])
    # P(phi)
    phi_values *= const['rad2deg']
    if phi_values.size == 1 or all(v - phi_values[0] < threshold for v in phi_values):
        phi_axis = np.arange(0, 360, ang_increment)
    else:
        phi_axis = np.arange(np.amin(phi_values)-10, np.amax(phi_values)+10, ang_increment)
    if phi_weights != []:
        phi_prob = histogram(phi_values, bins=phi_axis, weights=phi_weights)
    else:
        phi_prob = histogram(phi_values, bins=phi_axis)
    plt.subplot(2, 4, 4)
    plot_grid(fig, phi_axis, phi_prob, [r'$\mathit{\varphi}$ $^\circ$', r'$\mathit{P(\varphi)}$ (arb. u.)'])
    # P(alpha)
    alpha_values *= const['rad2deg']
    if alpha_values.size == 1 or all(v - alpha_values[0] < threshold for v in alpha_values):
        alpha_axis = np.arange(0, 360, ang_increment)
    else:
        alpha_axis = np.arange(np.amin(alpha_values)-10, np.amax(alpha_values)+10, ang_increment)
    if alpha_weights != []:
        alpha_prob = histogram(alpha_values, bins=alpha_axis, weights=alpha_weights)
    else:
        alpha_prob = histogram(alpha_values, bins=alpha_axis)
    plt.subplot(2, 4, 5)
    plot_grid(fig, alpha_axis, alpha_prob, [r'$\mathit{\alpha}$ $^\circ$', r'$\mathit{P(\alpha)}$ (arb. u.)'])
    # P(beta) * sin(beta)
    beta_values *= const['rad2deg']
    if beta_values.size == 1 or all(v - beta_values[0] < threshold for v in beta_values):
        beta_axis = np.arange(0, 180, ang_increment)
    else:
        beta_axis = np.arange(np.amin(beta_values)-10, np.amax(beta_values)+10, ang_increment)
    if beta_weights != []:
        beta_prob = histogram(beta_values, bins=beta_axis, weights=beta_weights)
    else:
        beta_prob = histogram(beta_values, bins=beta_axis)
    plt.subplot(2, 4, 6)
    plot_grid(fig, beta_axis, beta_prob, [r'$\mathit{\beta}$ $^\circ$', r'$\mathit{P(\beta)}$ (arb. u.)'])
    # P(gamma)
    gamma_values *= const['rad2deg']
    if gamma_values.size == 1 or all(v - gamma_values[0] < threshold for v in gamma_values):
        gamma_axis = np.arange(0, 360, ang_increment)
    else:
        gamma_axis = np.arange(np.amin(gamma_values)-10, np.amax(gamma_values)+10, ang_increment)
    if gamma_weights != []:
        gamma_prob = histogram(gamma_values, bins=gamma_axis, weights=gamma_weights)
    else:
        gamma_prob = histogram(gamma_values, bins=gamma_axis)
    plt.subplot(2, 4, 7)
    plot_grid(fig, gamma_axis, gamma_prob, [r'$\mathit{\gamma}$ $^\circ$', r'$\mathit{P(\gamma)}$ (arb. u.)'])
    plt.tight_layout()
    plt.draw()