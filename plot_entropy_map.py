import pylab as pl
import matplotlib.gridspec as gridspec

selected_dims = [10, 20, 170, -1]

for p0 in [42, 21, 10.5]:
    fig = pl.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, len(selected_dims))
    d = pl.load("results/entropy_map_p0={}.npz".format(p0))
    to_plot = d["results"].mean(axis=2)
    print(to_plot.shape)
    ax = fig.add_subplot(gs[0, :])
    p = ax.matshow(to_plot)
    pl.colorbar(p)
    for (i, s) in enumerate(selected_dims):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(to_plot[:, i])
        ax.set_ylim(0, 1)
        print(to_plot[:, i])
    fig.savefig("results/plots/entropy_map_p0={}.pdf".format(p0))
