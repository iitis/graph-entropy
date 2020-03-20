from matplotlib import rc

rc("text", usetex=True)
import pylab as pl
import numpy as np


mtx_labels = [
    r"$\mathrm{Adjacency}$",
    r"$\mathrm{Laplacian}$",
    r"$\mathrm{Norm. Laplacian}$",
]

model_labels = [r"$\mathrm{ER}$", r"$\mathrm{WS}$", r"$\mathrm{BA}$", r"$\mathrm{CL}$"]

fig, ax = pl.subplots(4, 3, figsize=(4, 3), sharex=True, sharey=True)
fig2, ax2 = pl.subplots(2, 2, figsize=(4, 3), sharex=True, sharey=True)
for i, model in enumerate(["er_p=0.4", "ws_k=4_p=0.6", "ba_m=2_k=2", "cl_m=2"]):
    for j, mtx in enumerate(["adjacency", "laplacian", "normalized_laplacian"]):
        try:
            if mtx == "adjacency":
                postfix = "_2"
            else:
                postfix = ""
            d = pl.load(
                "results/detail/detail_check_{}_{}{}.npz".format(model, mtx, postfix)
            )
            to_plot = d["result"].mean(axis=1)
            nrange = d["nrange"]
            color = "b"
        except FileNotFoundError:
            nrange = np.arange(10, 4000, 10)
            to_plot = np.log(nrange)
            color = "magenta"
        ax[i, j].plot(nrange, to_plot, color=color)
        ax[i, j].set_xticks([0, 2000, 4000])
        ax[i, j].set_xticklabels([r"$0$", r"$2000$", r"$4000$"])
        if i == 3 and j == 1:
            ax[i, j].set_xlabel(r"$n$")
        if j == 0 and i == 2:
            ax[i, j].set_ylabel(r"$S(\rho)$")
            ax[i, j].yaxis.set_label_coords(-0.4, 1.02)

        if i == 0:
            ax[i, j].set_title(r"{}".format(mtx_labels[j]))

        if j == 2:
            ax[i, j].set_ylabel(model_labels[i], rotation=0)
            ax[i, j].yaxis.set_label_coords(1.2, 0.55)
            ax[i, j].yaxis.set_label_position("right")
        if mtx == "normalized_laplacian":
            i2 = i // 2
            j2 = i % 2
            ax2[i2, j2].plot(nrange, np.log(nrange) - to_plot, color=color)
            ax2[i2, j2].set_title(r"{}".format(model_labels[i]))

fig.savefig("results/plots-paper/all-plots.pdf", bbox_inches="tight")
fig2.savefig("results/plots-paper/normed-laplacian.pdf", bbox_inches="tight")


fig, ax = pl.subplots(nrows=3, figsize=(4, 3), sharex=True)
for (i, p0) in enumerate([10.5, 21, 42]):
    d = pl.load("results/entropy_map_p0={}.npz".format(p0))
    to_plot = d["results"].mean(axis=2)
    p = ax[i].matshow(to_plot, vmin=0, vmax=1)

    ax[i].set_title("$p_0={}$".format(p0))
    ax[i].set_yticks([])
    ax[i].set_ylabel(r"$\tau$", labelpad=8)
    ax[i].tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    eps = d["eps"]
    a = d["arange"][0]
    b = d["brange"][0]
    inva = 19 * (b - a + a * b * eps) / (b - a + 2 * a * b * eps)
    invb = 19 * a * b * eps / (b - a + 2 * a * b * eps)
    print(inva, invb)
    ax[i].tick_params(axis="y", which="both", rotation=90, labelleft=False)
    ax[i].axhline(inva, color="white", lw=0.3)
    ax[i].axhline(invb, color="red", lw=0.5)
    ax[i].set_yticks([invb, inva])
    ax[i].set_yticklabels([r"$\frac{1}{b}$", r"$\frac{1}{a}$"])
    ax[i].set_ylim(19.5, -0.5)
    ax[i].text(-8, invb - 1, r"$\frac{1}{b}$")
    ax[i].text(-8, inva + 12, r"$\frac{1}{a}$")
cbaxes = fig.add_axes([0.94, 0.134, 0.03, 0.72])

i = 2
ax[i].tick_params(
    axis="x", which="both", bottom=True, top=False, labelbottom=True, labeltop=False
)
ax[i].set_xticks([0, 49, 99, 149, 199])
ax[i].set_xlabel("$n$")
ax[i].set_xticklabels(["$10$", "$500$", "$1000$", "$1500$", "$2000$"])

pl.colorbar(p, cax=cbaxes)
pl.subplots_adjust(hspace=1.0)
fig.savefig("results/plots-paper/entropy_map.pdf".format(p0), bbox_inches="tight")

param_labels = {"er": "p=", "ws": "p=", "ba": "m="}

taurange = np.logspace(-2, 4, 100)
fig, ax = pl.subplots(3, 3, figsize=(4, 3), sharex=True)
for (i, model) in enumerate(["er", "ws", "ba"]):
    for (j, mtx) in enumerate(["adjacency", "laplacian", "normalized_laplacian"]):
        d = pl.load("results/phase_transition/{}_{}.npz".format(model, mtx))
        n = d["n"]
        to_plot = d["result"].mean(axis=2)
        ax[i, j].plot(taurange, to_plot / np.log(n))
        ax[i, j].set_xscale("log")
        if i == 0:
            ax[i, j].set_title(r"{}".format(mtx_labels[j]))
        if i == 1 and j == 0:
            ax[i, j].set_ylabel("$S(\\rho)/ \\log n$")
        if j == 2:
            ax[i, j].set_ylabel(model_labels[i], rotation=0)
            ax[i, j].yaxis.set_label_coords(1.2, 0.55)
            ax[i, j].yaxis.set_label_position("right")
            if model == "er":
                labels = [
                    "${}{}$".format(param_labels[model], pp) for pp in d["params"]
                ]
            elif model == "ws":
                labels = [
                    "${}{}$".format(param_labels[model], pp[1]) for pp in d["params"]
                ]
            elif model == "ba":
                labels = [
                    "${}{}$".format(param_labels[model], pp[0]) for pp in d["params"]
                ]
            ax[i, j].legend(loc=0, bbox_to_anchor=(2.5, 1.0), labels=labels)
        if i == 2 and j == 1:
            ax[i, j].set_xlabel("$n$")
fig.savefig("results/plots-paper/phase-transition.pdf", bbox_inches="tight")

model_properties = {
    "ws": ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "p=", (0, 0)),
    "er": ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "p=", (0, 1)),
    "ba": ([1, 2, 3], "m=", (1, 0)),
    "cl": ([1, 2, 3], "m=", (1, 1)),
}

fig, ax = pl.subplots(3, 3, figsize=(4, 3), sharex=True, sharey=True)
for (i, model) in enumerate(["ws", "ba", "cl"]):
    for j, mtx in enumerate(["adjacency", "laplacian", "normalized_laplacian"]):
        model_param, param_label, (k, l) = model_properties[model]
        for (k, param) in enumerate(model_param):
            d = pl.load(
                "results/shift/{}_{}{}_{}.npz".format(model, param_label, param, mtx)
            )
            to_plot = d["result"].mean(axis=1)
            nrange = d["nrange"]
            ax[i, j].plot(
                nrange,
                np.log(nrange) - to_plot,
                label="${}{}$".format(param_label, param),
            )
        if i == 0:
            ax[i, j].set_title(r"{}".format(mtx_labels[j]))
        if j == 2:
            ax[i, j].set_ylabel(model_labels[i + 1], rotation=0)
            ax[i, j].yaxis.set_label_coords(1.2, 0.55)
            ax[i, j].yaxis.set_label_position("right")
        if i == 0 and j == 0:
            ax[i, j].legend(bbox_to_anchor=(3.475, 2.3), ncol=3)
        elif i == 2 and j == 2:
            ax[i, j].legend(bbox_to_anchor=(0.95, -0.6), ncol=3)
        if i == 2 and j == 1:
            ax[i, j].set_xlabel("$n$")
            ax[i, j].xaxis.set_label_coords(0.5, -0.4)

        if i == 1 and j == 0:
            ax[i, j].set_ylabel("$\\log n - S(\\rho)$")

fig.subplots_adjust(hspace=0.3)
fig.savefig("results/plots-paper/shift.pdf", bbox_inches="tight")
