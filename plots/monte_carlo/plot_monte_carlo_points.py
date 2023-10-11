import numpy as np
import plots.set_matplotlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 16})
from mathematics.histogram import histogram
from supplement.definitions import const


def plot_monte_carlo_points(
    r_values, xi_values, phi_values, alpha_values, beta_values, gamma_values, j_values, filename = "parameter_distributions.png"
    ):   
    """Plot integration grids."""            
    fig = plt.figure(figsize=(18, 9), facecolor="w", edgecolor="w")
    # P(r)
    if r_values != []:
        r_v, r_p = compute_distribution(r_values, 0, 80, 0.01)
        plt.subplot(2, 4, 1)
        plot_distribution(fig, r_v, r_p, [r"$\mathit{r}$ (nm)", r"$\mathit{P(r)}$ (arb. u.)"])
    # P(xi)
    if xi_values != []:
        xi_v, xi_p = compute_distribution(xi_values, -2 * np.pi, 2 * np.pi, np.pi / 1800)
        plt.subplot(2, 4, 2)
        plot_distribution(fig, const["rad2deg"] * xi_v, xi_p, [r"$\mathit{\xi}$ $^\circ$", r"$\mathit{P(\xi)}$ (arb. u.)"])
    # P(phi)
    if phi_values != []:
        phi_v, phi_p = compute_distribution(phi_values, -2 * np.pi, 2 * np.pi, np.pi / 1800)
        plt.subplot(2, 4, 3)
        plot_distribution(fig, const["rad2deg"] * phi_v, phi_p, [r"$\mathit{\varphi}$ $^\circ$", r"$\mathit{P(\varphi)}$ (arb. u.)"])
    # P(alpha)
    if alpha_values != []:
        alpha_v, alpha_p = compute_distribution(alpha_values, -2 * np.pi, 2 * np.pi, np.pi / 1800)
        plt.subplot(2, 4, 4)
        plot_distribution(fig, const["rad2deg"] * alpha_v, alpha_p, [r"$\mathit{\alpha}$ $^\circ$", r"$\mathit{P(\alpha)}$ (arb. u.)"])
    # P(beta)
    if beta_values != []:
        beta_v, beta_p = compute_distribution(beta_values, -2 * np.pi, 2 * np.pi, np.pi / 1800)
        plt.subplot(2, 4, 5)
        plot_distribution(fig, const["rad2deg"] * beta_v, beta_p, [r"$\mathit{\beta}$ $^\circ$", r"$\mathit{P(\beta)}$ (arb. u.)"])
    # P(gamma)
    if gamma_values != []:
        gamma_v, gamma_p = compute_distribution(gamma_values, -2 * np.pi, 2 * np.pi, np.pi / 1800)
        plt.subplot(2, 4, 6)
        plot_distribution(fig, const["rad2deg"] * gamma_v, gamma_p, [r"$\mathit{\gamma}$ $^\circ$", r"$\mathit{P(\gamma)}$ (arb. u.)"])
    # P(J)
    if j_values != []:
        j_v, j_p = compute_distribution(j_values, -20, 20, 0.01)
        plt.subplot(2, 4, 7)
        plot_distribution(fig, j_v, j_p, [r"$\mathit{J}$ (MHz)", r"$\mathit{P(J)}$ (arb. u.)"])
    plt.tight_layout()
    fig.savefig(filename, format = "png", dpi = 600)
    plt.close(fig)


def compute_distribution(points, minimum, maximum, increment):
    values = np.arange(minimum, maximum + increment, increment)
    probs = histogram(points, bins = values)
    probs = probs / np.amax(probs)
    return values, probs


def plot_distribution(
    fig, x, y, axes_labels = ["Parameter (arb. u.)", "Relative probability (arb. u.)"]
    ):
    """ Plot an integration grid """ 
    y = y / np.amax(y)
    axes = fig.gca()
    axes.plot(x, y, "k-")
    axes.set_xlim(np.amin(x), np.amax(x))
    axes.set_xlabel(axes_labels[0])
    axes.set_ylabel(axes_labels[1])