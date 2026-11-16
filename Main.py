"""Main file for running PyWake-Ellipsys simulations"""

# Import
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from matplotlib.animation import FuncAnimation, PillowWriter


# Import data
ds = xr.open_dataset(r"LES\Data\offset00deg\Flowpos06.nc")


# Define parameters
D = 282
R = D / 2
U_ref = 9.5


def plot_instant_wake(t):
    # Create normalized coordinates
    U = ds["U"].isel(t=t, z=0) / U_ref
    U_norm = U.assign_coords(
        x=U.x / D,
        y=U.y / D
    )

    plt.figure(figsize=(10,4))
    U_norm.plot(cmap="plasma")

    # Turbine locations
    for x_turb in [0, 7, 14, 21]:
        plt.plot(
            [x_turb, x_turb],
            [-0.5, 0.5],
            color="black",
            linewidth=3
        )

    plt.xlabel("x/D")
    plt.ylabel("y/D")
    plt.title(f"U/Uref at t={float(ds.t[t]):.1f} s")
    plt.show()

def plot_wake_animation(start_t, end_t):

    # Default step size
    step=1

    # Preload all required data
    print("Loading data into memory...")
    U_data = (
        ds["U"]
        .isel(t=slice(start_t, end_t, step), z=0)
        .values / U_ref
    )

    times = ds.t.isel(t=slice(start_t, end_t, step)).values

    print(f"Loaded {len(U_data)} frames.")

    fig, ax = plt.subplots(figsize=(10,4))

    im = ax.imshow(
        U_data[0],
        origin="lower",
        cmap="plasma",
        aspect="auto",
        extent=[
            float(ds.x.min()/D),
            float(ds.x.max()/D),
            float(ds.y.min()/D),
            float(ds.y.max()/D)
        ],
        vmin=0,
        vmax=1.1
    )

    plt.colorbar(im, ax=ax, label=r"$U/U_{ref}$")

    for x_turb in [0, 7, 14, 21]:
        ax.plot([x_turb, x_turb], [-0.5, 0.5], "k", lw=3)

    pbar = tqdm(total=len(U_data), desc="Rendering")

    def update(i):
        im.set_data(U_data[i])
        ax.set_title(f"LES wake field, t = {times[i]:.1f} s")
        pbar.update(1)
        return [im]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(U_data),
        blit=True
    )

    anim.save(
        "wake_animation.gif",
        writer="ffmpeg",
        fps=20
    )

    pbar.close()
    plt.close(fig)

    print("Saved wake_animation.mp4")

def extract_velocity_profile(x_pos, t_start, t_end):

    profile = (
        ds["U"]
        .isel(t=slice(t_start, t_end), z=0)
        .mean("t")
        .sel(x=x_pos, method="nearest")
    )

    profile = profile/U_ref

    return (
        profile.y.values,
        profile.values,
        float(profile.x)
    )

def plot_velocity_profile(x_pos, t_start, t_end,):

    y, U, actual_x = extract_velocity_profile(
        x_pos,
        t_start,
        t_end,
    )

    plt.figure(figsize=(8,4))

    plt.plot(U, y/D, lw=2)

    plt.axhspan(
        -0.5,
        0.5,
        color="black",
        alpha=0.15,
        label="Rotor"
    )

    plt.ylabel("y/D")
    plt.ylim(-2, 2)
    plt.xlabel(r"$U/U_{ref}$")
    plt.xlim(0.4, 1.1)

    plt.title(
        f"x = {actual_x/D:.2f}D, "
        f"timesteps {t_start}-{t_end}"
    )

    plt.grid(True)
    plt.legend()
    plt.show()

def plot_mean_wake(t_start, t_end):
    """
    Plot time-averaged wake field between t_start and t_end.
    """

    U_mean = (
        ds["U"]
        .isel(t=slice(t_start, t_end), z=0)
        .mean("t")
        / U_ref
    )

    plt.figure(figsize=(10,4))

    U_mean_norm = U_mean.assign_coords(
        x=U_mean.x / D,
        y=U_mean.y / D
    )

    U_mean_norm.plot(
        cmap="plasma",
        vmin=0,
        vmax=1.1,
        cbar_kwargs={"label": r"$U/U_{ref}$"}
    )

    # Turbines
    for x_turb in [0, 7, 14, 21]:
        plt.plot(
            [x_turb, x_turb],
            [-0.5, 0.5],
            color="black",
            linewidth=3
        )

    plt.xlabel("x/D")
    plt.ylabel("y/D")
    plt.title(
        f"Time-averaged wake\n"
        f"Timesteps {t_start}-{t_end}"
    )

    plt.tight_layout()
    plt.savefig("mean_wake.png", dpi=600)
    plt.show()

def save_velocity_profiles(t_start, t_end, dx_D=0.1):

    start_D = np.ceil((float(ds.x.min())/D)*10)/10
    end_D   = np.floor((float(ds.x.max())/D)*10)/10

    positions_D = np.arange(
        start_D,
        end_D + dx_D/2,
        dx_D
    )

    os.makedirs("velocity_profiles", exist_ok=True)

    U_mean = (
        ds["U"]
        .isel(t=slice(t_start, t_end), z=0)
        .mean("t")
        / U_ref
    )

    for xD in positions_D:

        # EXACT position via interpolation
        profile = U_mean.interp(x=xD * D)

        df = pd.DataFrame({
            "y_D": profile.y.values / D,
            "U_Uref": profile.values
        })

        if np.isclose(xD, 0):
            filename = f"0D_{t_end-t_start}s_mean.csv"

        elif xD < 0:
            filename = f"-{abs(xD):.1f}D_{t_end-t_start}s_mean.csv"

        else:
            filename = f"+{xD:.1f}D_{t_end-t_start}s_mean.csv"

        df.to_csv(
            os.path.join("velocity_profiles", filename),
            index=False
        )

        print(f"Saved {filename}")

    print("Finished.")

def plot_velocity_profiles(*positions):

    plt.figure(figsize=(6,6))

    for pos in positions:

        if np.isclose(pos, 0):
            filename = "0D_3600s_mean.csv"
            label = "0D"

        elif pos < 0:
            filename = f"-{abs(pos):.1f}D_3600s_mean.csv"
            label = f"-{abs(pos):.1f}D"

        else:
            filename = f"+{pos:.1f}D_3600s_mean.csv"
            label = f"+{pos:.1f}D"

        filepath = os.path.join(
            "LES",
            "extracted_velocity_profiles",
            filename
        )

        df = pd.read_csv(filepath)

        plt.plot(
            df["U_Uref"],
            df["y_D"],
            lw=2,
            label=label
        )

    plt.axhspan(
        -0.5,
        0.5,
        color="grey",
        alpha=0.15,
        label="Rotor"
    )

    plt.axhline(-0.5, color="k", ls="--")
    plt.axhline(0.5, color="k", ls="--")

    plt.xlabel(r"$U/U_{ref}$")
    plt.ylabel("y/D")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_instant_wake_centerline(t_idx):
    """
    Plot instantaneous wake field and the
    deficit-weighted wake centerline.
    """

    # Normalized velocity field
    U = ds["U"].isel(t=t_idx, z=0) / U_ref

    # Coordinates in D
    x_D = ds.x.values / D
    y_D = ds.y.values / D

    centerline_y = []

    for i in range(len(x_D)):

        profile = U.isel(x=i).values

        # Deficit
        deficit = 1 - profile

        # Avoid issues outside wake
        deficit = np.maximum(deficit, 0)

        if deficit.sum() > 0:
            yc = np.sum(y_D * deficit) / np.sum(deficit)
        else:
            yc = np.nan

        centerline_y.append(yc)

    centerline_y = np.array(centerline_y)

    # Heatmap
    U_plot = U.assign_coords(
        x=U.x / D,
        y=U.y / D
    )

    plt.figure(figsize=(12,4))

    U_plot.plot(
        cmap="plasma",
        vmin=0.3,
        vmax=1.1,
        cbar_kwargs={"label": r"$U/U_{ref}$"}
    )

    # Turbines
    for x_turb in [0, 7, 14, 21]:
        plt.plot(
            [x_turb, x_turb],
            [-0.5, 0.5],
            color="black",
            linewidth=3
        )

    # Wake centerline
    plt.plot(
        x_D,
        centerline_y,
        color="white",
        linewidth=2,
        label="Deficit COM"
    )

    plt.xlabel("x/D")
    plt.ylabel("y/D")
    plt.title(
        f"Wake centerline at t = {float(ds.t[t_idx]):.1f} s"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

def save_wake_centerline_gif(start_t, end_t, step=1,
                             filename="wake_centerline.gif",
                             fps=20):

    print("Loading data into memory...")

    U_data = (
        ds["U"]
        .isel(t=slice(start_t, end_t, step), z=0)
        .values / U_ref
    )

    times = ds.t.isel(
        t=slice(start_t, end_t, step)
    ).values

    x_D = ds.x.values / D
    y_D = ds.y.values / D

    print(f"Loaded {len(U_data)} frames.")

    fig, ax = plt.subplots(figsize=(12,4))

    im = ax.imshow(
        U_data[0],
        origin="lower",
        cmap="plasma",
        aspect="auto",
        extent=[
            x_D.min(),
            x_D.max(),
            y_D.min(),
            y_D.max()
        ],
        vmin=0.3,
        vmax=1.1
    )

    plt.colorbar(
        im,
        ax=ax,
        label=r"$U/U_{ref}$"
    )

    # turbines
    for x_turb in [0, 7, 14, 21]:
        ax.plot(
            [x_turb, x_turb],
            [-0.5, 0.5],
            color="black",
            linewidth=3
        )

    # centerline object
    centerline, = ax.plot(
        [],
        [],
        color="white",
        linewidth=2,
        label="Deficit COM"
    )

    ax.legend()

    pbar = tqdm(
        total=len(U_data),
        desc="Rendering GIF"
    )

    def update(i):

        U = U_data[i]

        centerline_y = []

        for j in range(U.shape[1]):  # loop over x

            profile = U[:, j]

            deficit = np.maximum(
                1 - profile,
                0
            )

            # Optional threshold
            deficit[deficit < 0.05] = 0

            if deficit.sum() > 0:
                yc = np.sum(
                    y_D * deficit
                ) / np.sum(deficit)
            else:
                yc = np.nan

            centerline_y.append(yc)

        centerline_y = np.array(centerline_y)

        im.set_data(U)

        centerline.set_data(
            x_D,
            centerline_y
        )

        ax.set_title(
            f"t = {times[i] - times[0]:.0f} s"
        )

        pbar.update(1)

        return [im, centerline]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(U_data),
        blit=True
    )

    anim.save(
        filename,
        writer=PillowWriter(fps=fps)
    )

    pbar.close()
    plt.close(fig)

    print(f"Saved {filename}")

save_wake_centerline_gif(
    start_t=0,
    end_t=3600,
)