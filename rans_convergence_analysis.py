import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


files = [
    r"RANS\data\0yaw0deg_cd10_e3.nc",
    r"RANS\data\0yaw0deg_cd10_e4.nc",
    r"RANS\data\0yaw0deg_cd10_e5.nc",
]

labels = [
    r"Residual = $10^{-3}$",
    r"Residual = $10^{-4}$",
    r"Residual = $10^{-5}$",
]


def plot_profile_comparison(
    files,
    labels,
    zPlot,
    D,
    variable="U",
    x_locations=[-1, 8, 15, 22, 29],
):

    fig, axes = plt.subplots(
        1,
        len(x_locations),
        figsize=(10, 4),
        sharey=True,
    )

    colors = ["k", "tab:blue", "tab:red"]

    for file, label, color in zip(files, labels, colors):

        ds = xr.open_dataset(file)

        x = ds.x.values / D
        y = ds.y.values / D

        field = ds[variable].interp(z=zPlot).values

        for ax, x_target in zip(axes, x_locations):

            ix = np.argmin(np.abs(x - x_target))

            profile = field[ix, :]

            ax.plot(
                profile,
                y,
                lw=2,
                color=color,
                label=label,
            )

            ax.set_title(fr"$x/D={x_target} [-]$")
            ax.grid(True)

    axes[0].set_ylabel(r"$y/D [-]$")

    for ax in axes:
        ax.set_xlabel(r"$U/U_{\mathrm{ref}}\;[-]$")

    axes[-1].legend()

    plt.tight_layout()
    plt.show()

plot_profile_comparison(
    files,
    labels,
    zPlot=170,
    D=284,
)