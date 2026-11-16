import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.style.use("classic")

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1


def plot_netCDF_flowvar_flatterrain(
    zPlot,
    D,
    infile=r"data\flowdata_basecase10.nc",
    cmap="plasma",
    dimensionless=True,
    variables=None,
    interpolate=True,
):
    data = xr.open_dataset(infile)

    flowvars = list(data.data_vars) if variables is None else variables

    x = data.x.values
    y = data.y.values

    if dimensionless:
        X, Y = np.meshgrid(x / D, y / D)
        xlabel = r"$x/D$"
        ylabel = r"$y/D$"
    else:
        X, Y = np.meshgrid(x, y)
        xlabel = r"$x\,[m]$"
        ylabel = r"$y\,[m]$"

    nvar = len(flowvars)

    Lx = x.max() - x.min()
    Ly = y.max() - y.min()

    figx = 10
    figy = figx * Ly / Lx * nvar

    fig, axes = plt.subplots(
        nvar,
        1,
        figsize=(figx, figy),
        sharex=True,
        sharey=True,
    )

    if nvar == 1:
        axes = [axes]

    cmap = plt.get_cmap(cmap)

    for ax, var in zip(axes, flowvars):

        method = "linear" if interpolate else "nearest"

        U = (
            data[var]
            .interp(z=zPlot, method=method)
            .values
        )

        pcm = ax.pcolormesh(
            X,
            Y,
            U.T,
            cmap=cmap,
            shading="auto",
        )

        units = data[var].attrs.get("units", "")

        ax.set_title(f"{var} [{units}]")
        ax.set_aspect("equal")
        ax.set_ylabel(ylabel)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)
        plt.colorbar(pcm, cax=cax)

    axes[-1].set_xlabel(xlabel)

    fig.tight_layout()
    fig.savefig("Wake.png", bbox_inches="tight")


if __name__ == "__main__":

    plot_netCDF_flowvar_flatterrain(
        zPlot=170.0,
        D=284.0,
        infile=r"RANS\data\flowdata_basecase10.nc",
        dimensionless=True,       # x/D, y/D
        # dimensionless=False,    # x [m], y [m]
        # variables=None,         # all variables
        variables=["U", "P"],   # selected variables
        interpolate=True,
    )
    