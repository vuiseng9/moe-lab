import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit


FONT_SIZE = 18
TITLE_SIZE = 20

FIT_X_MAX = 2.35
PLOT_X_MAX = 2.7

BASE_COLOR = "olive"
BASE_EDGE_COLOR = "darkolivegreen"
ALPHA_E_COLOR = "blue"
ALPHA_K_COLOR = "deeppink"
EXTRA_COLOR = "darkorange"


plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": 16,
})


# Points included in the diminishing-return fit
runtime = np.array(
    [2.31, 2.16, 2.09, 2.07, 1.99],
    dtype=float,
)

eval_loss = np.array(
    [1.0771, 1.0987, 1.1055, 1.1180, 1.1359],
    dtype=float,
)

alpha = [1, 2, 3, 4, 6]
l_vals = [768, 384, 256, 192, 128]


# Additional points excluded from the fit
alpha_e_point = (2.23, 1.0824)
alpha_k_point = (2.65, 1.0734)
extra_point = (2.491, 1.0998)


# Center the exponential fit for numerical stability
x0 = runtime.min()


def diminishing_return_curve(
    x: np.ndarray,
    y_inf: float,
    amplitude: float,
    decay_rate: float,
) -> np.ndarray:
    return y_inf + amplitude * np.exp(
        -decay_rate * (x - x0)
    )


# Fit only the olive ablation points
params, _ = curve_fit(
    diminishing_return_curve,
    runtime,
    eval_loss,
    p0=[1.06, 0.08, 8.0],
    bounds=(
        [0.0, 0.0, 0.0],
        [eval_loss.min(), 1.0, 100.0],
    ),
    maxfev=10_000,
)

x_fit = np.linspace(
    runtime.min(),
    FIT_X_MAX,
    500,
)

y_fit = diminishing_return_curve(
    x_fit,
    *params,
)


fig, ax = plt.subplots(figsize=(15, 8))


# Latent-factor ablation points
ax.scatter(
    runtime,
    eval_loss,
    s=110,
    color=BASE_COLOR,
    edgecolor=BASE_EDGE_COLOR,
    linewidth=0.8,
    zorder=3,
)


# Diminishing-return fit
ax.plot(
    x_fit,
    y_fit,
    linestyle=":",
    linewidth=3,
    color="lightgray",
    zorder=2,
)


# Baseline horizontal line
baseline_y = eval_loss[0]

ax.axhline(
    baseline_y,
    linestyle="--",
    linewidth=2,
    color=BASE_COLOR,
    alpha=0.95,
    zorder=1,
)

baseline_x = runtime[0]

ax.axvline(
    baseline_x,
    linestyle="--",
    linewidth=2,
    color=BASE_COLOR,
    alpha=0.95,
    zorder=1,
)

# Ablation-point annotations
for x, y, a, l_val in zip(
    runtime,
    eval_loss,
    alpha,
    l_vals,
):
    if a == 1:
        ax.annotate(
            f"α:{a}, l:{l_val} (baseline)",
            (x, y),
            xytext=(12, -14),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=FONT_SIZE,
        )
    else:
        ax.annotate(
            f"α:{a}, l:{l_val}",
            (x, y),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE,
        )


# Additional optimization points
ax.scatter(
    *alpha_e_point,
    marker="*",
    s=280,
    color=ALPHA_E_COLOR,
    zorder=4,
)

ax.scatter(
    *alpha_k_point,
    marker="*",
    s=280,
    color=ALPHA_K_COLOR,
    zorder=4,
)

ax.scatter(
    *extra_point,
    marker="*",
    s=280,
    color=EXTRA_COLOR,
    zorder=4,
)


alpha_e_label = "α:3, l:256, αE:48, K:2"
alpha_k_label = "α:3, l:256, αE:48, αK:6"
extra_label = "α:3, l:256, E:16, αK:6"


# Arrow from α=3 ablation point to αE point
alpha3_idx = alpha.index(3)

alpha3_point = (
    runtime[alpha3_idx],
    eval_loss[alpha3_idx],
)

ax.annotate(
    "",
    xy=alpha_e_point,
    xytext=alpha3_point,
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 2.2,
        "color": ALPHA_E_COLOR,
        "shrinkA": 10,
        "shrinkB": 12,
    },
    zorder=5,
)


alpha_e_mid_x = (
    alpha3_point[0] + alpha_e_point[0]
) / 2

alpha_e_mid_y = (
    alpha3_point[1] + alpha_e_point[1]
) / 2

ax.text(
    alpha_e_mid_x,
    alpha_e_mid_y - 0.0035,
    "αE",
    ha="center",
    va="top",
    fontsize=FONT_SIZE,
    color=ALPHA_E_COLOR,
)


# Arrow from αE point to αK point
ax.annotate(
    "",
    xy=alpha_k_point,
    xytext=alpha_e_point,
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 2.2,
        "color": ALPHA_K_COLOR,
        "shrinkA": 14,
        "shrinkB": 14,
    },
    zorder=5,
)


alpha_k_mid_x = (
    alpha_e_point[0] + alpha_k_point[0]
) / 2

alpha_k_mid_y = (
    alpha_e_point[1] + alpha_k_point[1]
) / 2

ax.text(
    alpha_k_mid_x,
    alpha_k_mid_y + 0.0012,
    "αK",
    ha="center",
    va="bottom",
    fontsize=FONT_SIZE,
    color=ALPHA_K_COLOR,
)


# Custom legend
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=10,
        markerfacecolor=BASE_COLOR,
        markeredgecolor=BASE_EDGE_COLOR,
        label="ablate latent factor, α",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markersize=16,
        markerfacecolor=ALPHA_E_COLOR,
        markeredgecolor=ALPHA_E_COLOR,
        label=alpha_e_label,
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markersize=16,
        markerfacecolor=ALPHA_K_COLOR,
        markeredgecolor=ALPHA_K_COLOR,
        label=alpha_k_label,
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markersize=16,
        markerfacecolor=EXTRA_COLOR,
        markeredgecolor=EXTRA_COLOR,
        label=extra_label,
    ),
]

ax.legend(
    handles=legend_handles,
    loc="upper right",
    frameon=True,
)


# Title and axes
ax.set_title(
    r"$\bf{LatentMoE\ Optimization}$"
    "\n"
    "on MoE E=16, K=2, D=786, Dff=1024",
    pad=14,
)

ax.set_xlabel(
    "Runtime (hr)\n"
    "← better"
)

ax.set_ylabel(
    "Eval loss\n"
    "↓ better",
    rotation=0,
    labelpad=70,
    va="center",
)

ax.set_xlim(right=PLOT_X_MAX)
ax.grid(True, alpha=0.3)
ax.margins(x=0.1, y=0.12)

fig.tight_layout()


output_path = "runtime_vs_eval_loss.png"

fig.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)

fig.savefig("ablations_latentmoe.png", dpi=200, bbox_inches="tight")

# plt.show()