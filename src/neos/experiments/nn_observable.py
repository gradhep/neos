__all__ = (
    "generate_data",
    "make_model",
    "nn_summary_stat",
    "plot",
    "plot_setup",
    "first_epoch",
    "last_epoch",
    "per_epoch",
)

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import relaxed
from celluloid import Camera
from jax.random import PRNGKey, multivariate_normal


def generate_data(
    rng=0,
    num_points=10000,
    sig_mean=(-1, 1),
    bup_mean=(2.5, 2),
    bdown_mean=(-2.5, -1.5),
    b_mean=(1, -1),
):
    sig = multivariate_normal(
        PRNGKey(rng),
        jnp.asarray(sig_mean),
        jnp.asarray([[1, 0], [0, 1]]),
        shape=(num_points,),
    )
    bkg_up = multivariate_normal(
        PRNGKey(rng),
        jnp.asarray(bup_mean),
        jnp.asarray([[1, 0], [0, 1]]),
        shape=(num_points,),
    )
    bkg_down = multivariate_normal(
        PRNGKey(rng),
        jnp.asarray(bdown_mean),
        jnp.asarray([[1, 0], [0, 1]]),
        shape=(num_points,),
    )

    bkg_nom = multivariate_normal(
        PRNGKey(rng),
        jnp.asarray(b_mean),
        jnp.asarray([[1, 0], [0, 1]]),
        shape=(num_points,),
    )
    return sig, bkg_nom, bkg_up, bkg_down


def make_model(s, b_nom, b_up, b_down, validate=False):
    m = {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": s,
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None},
                        ],
                    },
                    {
                        "name": "background",
                        "data": b_nom,
                        "modifiers": [
                            {
                                "name": "correlated_bkg_uncertainty",
                                "type": "histosys",
                                "data": {"hi_data": b_up, "lo_data": b_down},
                            },
                        ],
                    },
                ],
            },
        ],
    }
    return pyhf.Model(m, validate=validate)


def nn_summary_stat(
    pars,
    data,
    nn,
    bandwidth,
    bins,
    reflect=False,
    sig_scale=2,
    bkg_scale=10,
    LUMI=10,
    return_preds=False,
    use_kde=True,
    use_list=False,
):
    s_data, b_nom_data, b_up_data, b_down_data = data

    preds = nn_s, nn_b_nom, nn_b_up, nn_b_down = (
        nn(pars, s_data).ravel(),
        nn(pars, b_nom_data).ravel(),
        nn(pars, b_up_data).ravel(),
        nn(pars, b_down_data).ravel(),
    )

    num_points = len(s_data)
    if use_kde:
        yields = s, b_nom, b_up, b_down = [
            relaxed.hist(nn_s, bins, bandwidth, reflect_infinities=reflect)
            * sig_scale
            / num_points
            * LUMI,
            relaxed.hist(nn_b_nom, bins, bandwidth, reflect_infinities=reflect)
            * bkg_scale
            / num_points
            * LUMI,
            relaxed.hist(nn_b_up, bins, bandwidth, reflect_infinities=reflect)
            * bkg_scale
            / num_points
            * LUMI,
            relaxed.hist(
                nn_b_down,
                bins,
                bandwidth,
                reflect_infinities=reflect,
            )
            * bkg_scale
            / num_points
            * LUMI,
        ]
    elif not use_kde:
        yields = [
            np.histogram(nn_s, bins=bins)[0] * sig_scale / num_points * LUMI,
            np.histogram(nn_b_nom, bins=bins)[0] * bkg_scale / num_points * LUMI,
            np.histogram(nn_b_up, bins=bins)[0] * bkg_scale / num_points * LUMI,
            np.histogram(nn_b_down, bins=bins)[0] * bkg_scale / num_points * LUMI,
        ]
        if use_list:
            yields = [y.tolist() for y in yields]
    else:
        raise ValueError("use_kde must be True or False")
    if return_preds:
        return yields, preds
    return yields


def make_kde(data, bw):
    @jax.jit
    def get_kde(x):
        return jnp.mean(
            jsp.stats.norm.pdf(x, loc=data.reshape(-1, 1), scale=bw), axis=0
        )

    return get_kde


def bar_plot(
    ax, data, colors=None, total_width=0.8, single_width=1, legend=True, bins=None
):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (_, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(
                x + x_offset,
                y,
                width=bar_width * single_width,
                color=colors[i % len(colors)],
            )

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    labels = [f"[{a:.1g},{b:.1g}]" for a, b in zip(bins[:-1], bins[1:])]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    return bars, data.keys()


def plot(
    network,
    axs,
    axins,
    metrics,
    maxN,
    this_batch,
    epoch_grid,
    nn,
    bins,
    bandwidth,
    batch_num,
    legend=False,
    reflect=False,
    histlim=55,
):
    if "Likelihood scan" in axs:
        ax = axs["Likelihood scan"]
        import cabinetry

        model = make_model(*metrics["yields"])
        bonly_pars = (
            jnp.asarray(model.config.suggested_init())
            .at[model.config.poi_index]
            .set(0.0)
            .tolist()
        )
        data_hf = model.expected_data(bonly_pars)
        scan_results = cabinetry.fit.scan(
            model, data_hf, "mu", par_bounds=[[-2, 10], [-2, 10]]
        )
        cabinetry.visualize.scan(scan_results, existing_ax=ax, legend=legend)
    if "Expected limits" in axs:
        # if batch_num != 0:
        ax = axs["Expected limits"]
        import cabinetry

        model = make_model(*metrics["yields"])
        bonly_pars = (
            jnp.asarray(model.config.suggested_init())
            .at[model.config.poi_index]
            .set(0.0)
            .tolist()
        )
        data_hf = model.expected_data(bonly_pars)

        limit_results = cabinetry.fit.limit(model, data_hf, maxiter=1000)
        cabinetry.visualize.limit(limit_results, existing_ax=ax, legend=legend)

    if "Data space" in axs:
        ax = axs["Data space"]
        g = np.mgrid[-5:5:101j, -5:5:101j]
        if jnp.inf in bins:
            levels = bins[1:-1]  # infinite
        else:
            levels = bins
        ax.contourf(
            g[0],
            g[1],
            nn(network, np.moveaxis(g, 0, -1)).reshape(101, 101, 1)[:, :, 0],
            levels=levels,
            cmap="binary",
        )
        ax.contour(
            g[0],
            g[1],
            nn(network, np.moveaxis(g, 0, -1)).reshape(101, 101, 1)[:, :, 0],
            colors="w",
            levels=levels,
        )
        sig, bkg_nom, bkg_up, bkg_down = this_batch
        # should definitely not have to repeat this every time lmao
        ax.scatter(sig[:, 0], sig[:, 1], alpha=0.3, c="C9", label="signal")
        ax.scatter(
            bkg_up[:, 0],
            bkg_up[:, 1],
            alpha=0.1,
            c="orangered",
            marker=6,
            label="bkg up",
        )
        ax.scatter(
            bkg_down[:, 0],
            bkg_down[:, 1],
            alpha=0.1,
            c="gold",
            marker=7,
            label="bkg down",
        )
        ax.scatter(bkg_nom[:, 0], bkg_nom[:, 1], alpha=0.3, c="C1", label="bkg")

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if legend:
            ax.legend(fontsize="x-small", loc="upper right", fancybox=True)

        # ax.axis("off")
    x_grid = epoch_grid[: batch_num + 1]
    if "Losses" in axs:
        ax = axs["Losses"]
        ax.plot(
            epoch_grid[: batch_num + 1],
            metrics["loss"],
            c="C9",
            linewidth=2.0,
            label=r"train",
        )
        ax.plot(
            epoch_grid[: batch_num + 1],
            metrics["test_loss"],
            c="C4",
            linewidth=2.0,
            label=r"test",
        )
        ax.set_yscale("log")
        ax.set_xlim(0, maxN)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"loss value")
        if legend:
            ax.legend(fontsize="x-small", loc="upper right", fancybox=True)

    if "Metrics" in axs:
        ax = axs["Metrics"]
        ax.plot(
            x_grid,
            metrics["1-pull_width**2"],
            c="slategray",
            linewidth=2.0,
            label=r"$(1-\sigma_{\mathsf{nuisance}})^2$",
            # linestyle=":"
        )
        ax.plot(
            x_grid,
            np.array(metrics["pull"]) ** 2,
            c="C2",
            linewidth=2.0,
            label=r"(nuisance pull)$^2$",
            # linestyle=':'
        )
        ax.plot(
            x_grid,
            metrics["mu_uncert"],
            c="steelblue",
            linewidth=2.0,
            label=r"$\sigma_\mu$",
        )
        ax.plot(x_grid, metrics["CLs"], c="C9", linewidth=2, label=r"$CL_s$")
        ax.set_ylim(1e-7, 1e-0)
        ax.set_xlim(0, maxN)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.set_ylabel(r"metric value (on test set)")
        if legend:
            ax.legend(fontsize="x-small", loc="upper right", fancybox=True)
    if "Histogram model" in axs:
        ax = axs["Histogram model"]
        s, b, bup, bdown = metrics["yields"]

        if jnp.inf in bins:
            noinf = bins[1:-1]
            bin_width = 1 / (len(noinf) - 1)
            centers = noinf[:-1] + np.diff(noinf) / 2.0
            centers = jnp.array([noinf[0] - bin_width, *centers, noinf[-1] + bin_width])

        dct = {
            "signal": s,
            "bkg up": bup,
            "bkg": b,
            "bkg down": bdown,
        }

        a, b = bar_plot(
            ax,
            dct,
            colors=["C9", "orangered", "C1", "gold"],
            total_width=0.8,
            single_width=1,
            legend=legend,
            bins=bins,
        )
        ax.set_ylabel("frequency")
        ax.set_xlabel("interval over nn output")
        ax.set_ylim(0, histlim)
        # ax.axis("off")

        if legend:
            # Draw legend if we need
            # ax.legend(a, b, fontsize="x-small")
            if jnp.inf in bins:
                width = jnp.diff(noinf)[0]
            else:
                width = jnp.diff(bins)[0]
            xlim = (
                [(width / 2) - (1.1 * bandwidth), (width / 2) + (1.1 * bandwidth)]
                if (width / 2) - bandwidth < 0
                else [-width / 3, width + width / 3]
            )
            axins.stairs([1], [0, width], color="C1", alpha=0.6)
            y = jnp.linspace(xlim[0], xlim[1], 300)
            demo = jsp.stats.norm.pdf(y, loc=width / 2, scale=bandwidth)
            axins.plot(
                y, demo / max(demo), color="C0", linestyle="dashed", label="kernel"
            )
            # draw two vertical lines at ((width/2)-bandwidth)/2 and ((width/2)+bandwidth)/2
            axins.vlines(
                [(width / 2) - bandwidth, (width / 2) + bandwidth],
                0,
                1,
                colors="black",
                linestyles="dotted",
                label=r"$\pm$bandwidth",
                alpha=0.9,
            )
            # write text in the middle of the vertical lines with the value of the bandwidth
            ratio = bandwidth / width
            axins.text(
                width / 2,
                -0.3,
                r"$\mathsf{\frac{bandwidth}{bin\,width}}=$" + f"{ratio:.2f}",
                ha="center",
                va="center",
                size="x-small",
                alpha=0.9,
            )

            axins.set_xlim(*xlim)

            handles, labels = a, list(b)  # ax.get_legend_handles_labels()
            handles1, labels1 = axins.get_legend_handles_labels()
            ax.legend(
                handles + handles1,
                labels + labels1,
                loc="upper right",
                fontsize="x-small",
                fancybox=True,
            )

    if "Nuisance pull" in axs:
        ax = axs["Nuisance pull"]

        pulls = metrics["pull"]
        pullerr = metrics["pull_width"]

        ax.set_ylabel(r"$(\theta - \hat{\theta})\,/ \Delta \theta$", fontsize=18)

        # draw the +/- 2.0 horizontal lines
        ax.hlines([-2, 2], -0.5, len(pulls) - 0.5, colors="black", linestyles="dotted")
        # draw the +/- 1.0 horizontal lines
        ax.hlines([-1, 1], -0.5, len(pulls) - 0.5, colors="black", linestyles="dashdot")
        # draw the +/- 2.0 sigma band
        ax.fill_between([-0.5, len(pulls) - 0.5], [-2, -2], [2, 2], facecolor="yellow")
        # drawe the +/- 1.0 sigma band
        ax.fill_between([-0.5, len(pulls) - 0.5], [-1, -1], [1, 1], facecolor="green")
        # draw a horizontal line at pull=0.0
        ax.hlines([0], -0.5, len(pulls) - 0.5, colors="black", linestyles="dashed")

        ax.scatter(range(len(pulls)), pulls, color="black")
        # and their uncertainties
        ax.errorbar(
            range(len(pulls)),
            pulls,
            color="black",
            xerr=0,
            yerr=pullerr,
            marker=".",
            fmt="none",
        )
    if "Example KDE" in axs:
        ax = axs["Example KDE"]
        b_data = bkg_nom
        d = np.array(nn(network, b_data).ravel().tolist())
        kde = make_kde(d, bandwidth)
        yields = b
        ls = [-1, 2]
        x = np.linspace(ls[0], ls[1], 300)
        db = jnp.array(jnp.diff(bins), float)  # bin spacing
        yields = yields / db / yields.sum(axis=0)  # normalize to bin width
        if jnp.inf in bins:
            pbins = [ls[0], *noinf, ls[1]]
        else:
            pbins = bins
        ax.stairs(yields, pbins, label="KDE hist", color="C1")
        if reflect:
            ax.plot(x, 2 * jnp.abs(kde(x)), label="KDE", color="C0")
        else:
            ax.plot(x, kde(x), label="KDE", color="C0")

        ax.set_xlim(*ls)

        # rug plot of the data
        ax.plot(
            d,
            jnp.zeros_like(d) - 0.01,
            "|",
            linewidth=3,
            alpha=0.4,
            color="black",
            label="data",
        )

        if legend:
            if jnp.inf in bins:

                width = jnp.diff(noinf)[0]
            else:
                width = jnp.diff(bins)[0]
            xlim = (
                [(width / 2) - (1.1 * bandwidth), (width / 2) + (1.1 * bandwidth)]
                if (width / 2) - bandwidth < 0
                else [-width / 3, width + width / 3]
            )
            axins.stairs([1], [0, width], color="C1")
            y = jnp.linspace(xlim[0], xlim[1], 300)
            demo = jsp.stats.norm.pdf(y, loc=width / 2, scale=bandwidth)
            axins.plot(
                y, demo / max(demo), color="C0", linestyle="dashed", label="kernel"
            )
            # draw two vertical lines at ((width/2)-bandwidth)/2 and ((width/2)+bandwidth)/2
            axins.vlines(
                [(width / 2) - bandwidth, (width / 2) + bandwidth],
                0,
                1,
                colors="black",
                linestyles="dotted",
                label=r"$\pm$bandwidth",
            )
            # write text in the middle of the vertical lines with the value of the bandwidth
            ratio = bandwidth / width
            axins.text(
                width / 2,
                -0.3,
                r"$\mathsf{\frac{bandwidth}{bin\,width}}=$" + f"{ratio:.2f}",
                ha="center",
                va="center",
                size="x-small",
            )

            axins.set_xlim(*xlim)

            handles, labels = ax.get_legend_handles_labels()
            handles1, labels1 = axins.get_legend_handles_labels()
            ax.legend(
                handles + handles1,
                labels + labels1,
                loc="upper right",
                fontsize="x-small",
                fancybox=True,
            )


def first_epoch(
    network,
    camera,
    axs,
    axins,
    metrics,
    maxN,
    this_batch,
    epoch_grid,
    batch_num,
    nn,
    bins,
    bandwidth,
    histlim,
    **kwargs,
):
    plot(
        axs=axs,
        axins=axins,
        network=network,
        metrics=metrics,
        maxN=maxN,
        this_batch=this_batch,
        batch_num=batch_num,
        epoch_grid=epoch_grid,
        nn=nn,
        bins=bins,
        bandwidth=bandwidth,
        legend=True,
        histlim=histlim,
    )
    plt.tight_layout()
    if camera is not None:
        camera.snap()
    return camera


def last_epoch(
    network,
    camera,
    axs,
    axins,
    metrics,
    maxN,
    this_batch,
    batch_num,
    epoch_grid,
    nn,
    bins,
    bandwidth,
    pipeline,
    histlim,
    **kwargs,
):
    plot(
        axs=axs,
        axins=axins,
        network=network,
        metrics=metrics,
        maxN=maxN,
        this_batch=this_batch,
        epoch_grid=epoch_grid,
        batch_num=batch_num,
        nn=nn,
        bins=bins,
        bandwidth=bandwidth,
        histlim=histlim,
        legend=False,
    )
    plt.tight_layout()
    if camera is not None:
        camera.snap()

    defaults = {
        "layout": [
            ["Data space", "Histogram model"],
            ["Losses", "Metrics"],
        ]
    }
    if pipeline.plot_kwargs is not None:
        for k in defaults:
            if k in pipeline.plot_kwargs:
                defaults[k] = pipeline.plot_kwargs[k]
    fig2, axs2 = plt.subplot_mosaic(defaults["layout"])

    for label, ax in axs2.items():
        ax.set_title(label, fontstyle="italic")
    if "Histogram model" in axs2:
        axins2 = axs2["Histogram model"].inset_axes([0.33, 0.79, 0.3, 0.2])
        axins2.axis("off")
    else:
        axins2 = None
    plot(
        axs=axs2,
        axins=axins2,
        network=network,
        metrics=metrics,
        maxN=maxN,
        this_batch=this_batch,
        nn=nn,
        epoch_grid=epoch_grid,
        batch_num=batch_num,
        bins=bins,
        bandwidth=bandwidth,
        legend=True,
        histlim=histlim,
    )
    plt.tight_layout()
    fig2.savefig(f"{pipeline.plot_name}")
    return camera


def per_epoch(
    network,
    camera,
    axs,
    axins,
    metrics,
    maxN,
    this_batch,
    epoch_grid,
    batch_num,
    nn,
    bins,
    bandwidth,
    histlim,
    **kwargs,
):
    plot(
        axs=axs,
        axins=axins,
        network=network,
        metrics=metrics,
        maxN=maxN,
        this_batch=this_batch,
        epoch_grid=epoch_grid,
        batch_num=batch_num,
        nn=nn,
        bins=bins,
        bandwidth=bandwidth,
        legend=False,
        histlim=histlim,
    )
    plt.tight_layout()
    if camera is not None:
        camera.snap()
    return camera


def plot_setup(pipeline):
    plt.style.use("default")

    defaults = {
        "layout": [
            ["Data space", "Histogram model"],
            ["Losses", "Metrics"],
        ]
    }
    if pipeline.plot_kwargs is not None:
        for k in defaults:
            if k in pipeline.plot_kwargs:
                defaults[k] = pipeline.plot_kwargs[k]

    plt.rcParams.update(
        {
            "axes.labelsize": 13,
            "axes.linewidth": 1.2,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.figsize": [10.0, 10.0],
            "font.size": 13,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 11,
            "legend.fancybox": True,
        }
    )

    plt.rc("figure", dpi=150)

    fig, axs = plt.subplot_mosaic(defaults["layout"])

    for label, ax in axs.items():
        ax.set_title(label, fontstyle="italic")
    # axs["Example KDE"].set_title("Example KDE (nominal bkg)", fontstyle="italic")
    if "Histogram model" in axs:
        axins = axs["Histogram model"].inset_axes([0.33, 0.79, 0.3, 0.2])
        axins.axis("off")
        axins_cpy = axins
    else:
        axins = axins_cpy = None
    ax_cpy = axs
    if pipeline.animate:
        camera = Camera(fig)
    else:
        camera = None
    plt.suptitle(pipeline.plot_title, fontsize="x-large")
    return dict(
        camera=camera, axs=axs, axins=axins, ax_cpy=ax_cpy, axins_cpy=axins_cpy, fig=fig
    )
