"""Main file for running PyWake-Ellipsys simulations"""

# Import
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from matplotlib.animation import FuncAnimation, PillowWriter
import glob
import matplotlib.cm as cm
from scipy.interpolate import RegularGridInterpolator

from analysis import WakeSimulation


# Import data
ds = xr.open_dataset(r"LES\Data\offset00deg\Flowpos06.nc")


# Define parameters
D = 282
R = D / 2
U_ref = 9.5

# Create wake simulation instance
#wake_sim = WakeSimulation(ds, D, U_ref)
#wake_sim.plot_instant_wake(t=0)

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

    # Time label inside top-right corner
    time_text = ax.text(
        0.98, 0.95,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color="black",
    )

    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$")

    plt.colorbar(im, ax=ax, label=r"$U/U_{0}$")

    for x_turb in [0, 7, 14, 21]:
        ax.plot([x_turb, x_turb], [-0.5, 0.5], "k", lw=3)

    pbar = tqdm(total=len(U_data), desc="Rendering")

    def update(i):
        im.set_data(U_data[i])
        time_text.set_text(f"t = {times[i]:.1f} s")
        pbar.update(1)
        return [im, time_text]

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
    #plt.show()

    df = U_mean_norm.to_dataframe(name="U_Uref").reset_index()

    df.to_csv(
        f"mean_wake_t{t_start}_{t_end}_long.csv",
        index=False
    )

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

def save_wake_centerline_gif(
        start_t,
        end_t,
        step=1,
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

    fig, ax = plt.subplots(figsize=(12, 4))

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

    for x_turb in [0, 7, 14, 21]:
        ax.plot(
            [x_turb, x_turb],
            [-0.5, 0.5],
            color="black",
            linewidth=3
        )

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

        for j in range(U.shape[1]):

            profile = U[:, j]

            deficit = np.maximum(
                1 - profile,
                0
            )

            # Remove weak deficits
            deficit[deficit < 0.10] = 0

            if deficit.sum() > 0:

                weights = deficit**2

                yc = np.sum(
                    y_D * weights
                ) / np.sum(weights)

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

def find_max_deficits(csv_file, sections):
    """
    sections = [
        (0, 7),
        (7, 14),
        (14, 21),
        (21, 28)
    ]
    """

    df = pd.read_csv(csv_file)

    results = []

    for i, (x_min, x_max) in enumerate(sections, start=1):

        region = df[
            (df["x"] >= x_min) &
            (df["x"] <= x_max)
        ].copy()

        region["deficit"] = 1 - region["U_Uref"]

        idx = region["deficit"].idxmax()
        row = region.loc[idx]

        results.append({
            "turbine": i,
            "x_D": row["x"],
            "y_D": row["y"],
            "U_Uref": row["U_Uref"],
            "deficit": row["deficit"]
        })

    return pd.DataFrame(results)

def plot_max_deficits(result):

    plt.figure(figsize=(6,4))

    plt.plot(
        result["turbine"],
        result["deficit"],
        "-o",
        linewidth=2,
        markersize=8
    )

    plt.xlabel("Turbine")
    plt.ylabel(r"Maximum deficit ($1-U/U_{ref}$)")
    plt.title("Maximum wake deficit behind each turbine")

    plt.xticks(result["turbine"])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_section_deficit_timeseries(sections):

    x_D = ds.x.values / D
    times = ds.t.values

    mean_velocities = np.zeros((len(times), len(sections)))

    print("\nSection information")
    print("-" * 60)

    for i, (x_min, x_max) in enumerate(sections):

        x_mask = (x_D >= x_min) & (x_D <= x_max)

        U_section = (
            ds["U"]
            .isel(z=0)
            .isel(x=x_mask)
            .values
            / U_ref
        )

        print(
            f"Section {i+1}: "
            f"{U_section.shape[2]} x-cells × "
            f"{U_section.shape[1]} y-cells = "
            f"{U_section.shape[1] * U_section.shape[2]} cells"
        )

        # Average velocity in the section at each timestep
        mean_velocities[:, i] = U_section.mean(axis=(1, 2))

    print("\nWake deficit statistics")
    print("-" * 60)

    for i, (x_min, x_max) in enumerate(sections):

        avg_u = mean_velocities[:, i].mean()

        # strongest average deficit encountered
        min_u = mean_velocities[:, i].min()

        avg_deficit = 1 - avg_u
        max_deficit = 1 - min_u

        print(
            f"Section {i+1} ({x_min}D to {x_max}D):\n"
            f"    Average deficit = {avg_deficit:.4f}\n"
            f"    Maximum deficit = {max_deficit:.4f}\n"
        )

    plt.figure(figsize=(10, 5))

    for i in range(len(sections)):

        deficit = 1 - mean_velocities[:, i]

        plt.plot(
            times - times[0],
            deficit,
            linewidth=1.5,
            label=f"Turbine {i+1}"
        )

    plt.xlabel("Time [s]")
    plt.ylabel(r"Section-averaged deficit ($1-U/U_{ref}$)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return times, mean_velocities

def compute_average_difference_from_60min():
    """
    Compute the average difference between 1-min mean and 60-min mean.
    """
    # Load reference (60 min average)
    df_60 = pd.read_csv(
        r"Results\60_min_average\mean_wake_t0_3600_long.csv"
    )

    files = glob.glob(
        r"Results\2_min_average\*.csv"
    )

    print(f"Found {len(files)} files")

    # Initialize accumulator
    avg_diff = np.zeros(len(df_60))

    # Loop through all files
    for file in files:

        print("Processing:", os.path.basename(file))

        df = pd.read_csv(file)

        # Difference from 60-min mean
        diff = np.abs(
            df["U_Uref"] - df_60["U_Uref"]
        )

        avg_diff += diff

    # Average over all files
    avg_diff /= len(files)

    # Save
    df_out = df_60.copy()
    df_out["avg_difference"] = avg_diff

    df_out.to_csv(
        r"Results\average_2min_difference_from_60min.csv",
        index=False
    )

    print("Saved average_2min_difference_from_60min.csv")

    heatmap = df_out.pivot(
        index="y",
        columns="x",
        values="avg_difference"
    )

    X, Y = np.meshgrid(
        heatmap.columns,
        heatmap.index
    )

    plt.figure(figsize=(12,6))

    cf = plt.contourf(
        X,
        Y,
        heatmap.values,
        levels=41,
        cmap="viridis"
    )

    plt.colorbar(
        cf,
        label=r"$\frac{\sum_i\left(U_{2\mathrm{min},i}-U_{60\mathrm{min}}\right)}{U_{\mathrm{ref}}}$"
    )

    plt.xlabel("x/D")
    plt.ylabel("y/D")
    plt.title("Mean Absolute Difference from 60-Minute Average")

    plt.tight_layout()
    plt.show()

def wake_centerline_les_60min_average():
    """
    Compute the wake centerline from the 60-min mean field.
    """
    # Load 60-min mean field
    df = pd.read_csv(
        r"Results\60_min_average\mean_wake_t0_3600_long.csv"
    )

    # Create heatmap
    heatmap = df.pivot(
        index="y",
        columns="x",
        values="U_Uref"
    )

    x_D = heatmap.columns.values
    y_D = heatmap.index.values

    U = heatmap.values

    # --------------------------------------------------
    # Compute wake centerline
    # --------------------------------------------------

    centerline_y = []

    for j in range(U.shape[1]):

        profile = U[:, j]

        deficit = np.maximum(
            1 - profile,
            0
        )

        # Same threshold as GIF
        deficit[deficit < 0.10] = 0

        if deficit.sum() > 0:

            weights = deficit**2

            yc = (
                np.sum(y_D * weights)
                / np.sum(weights)
            )

        else:
            yc = np.nan

        centerline_y.append(yc)

    centerline_y = np.array(centerline_y)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    X, Y = np.meshgrid(
        x_D,
        y_D
    )

    plt.figure(figsize=(12,6))

    cf = plt.contourf(
        X,
        Y,
        U,
        levels=100,
        cmap="plasma"
    )

    plt.colorbar(
        cf,
        label=r"$U/U_{ref}$"
    )

    # turbine locations
    for x_turb in [0, 7, 14, 21]:
        plt.plot(
            [x_turb, x_turb],
            [-0.5, 0.5],
            color="black",
            linewidth=3
        )

    # wake centerline
    plt.plot(
        x_D,
        centerline_y,
        color="white",
        linewidth=3,
        label="Wake centerline"
    )

    plt.xlabel("x/D")
    plt.ylabel("y/D")

    plt.legend()
    plt.tight_layout()
    plt.show()

def wake_centerline_les_60min_average_comparison_white_background():
    """
    Compute the wake centerline from the 60-min mean field.
    """
    # Load 60-min mean field
    df = pd.read_csv(
        r"Results\60_min_average\mean_wake_t0_3600_long.csv"
    )

    # Create heatmap
    heatmap = df.pivot(
        index="y",
        columns="x",
        values="U_Uref"
    )

    x_D = heatmap.columns.values
    y_D = heatmap.index.values

    U = heatmap.values

    # --------------------------------------------------
    # Compute wake centerline
    # --------------------------------------------------

    centerline_y = []

    for j in range(U.shape[1]):

        profile = U[:, j]

        deficit = np.maximum(
            1 - profile,
            0
        )

        # Same threshold as GIF
        deficit[deficit < 0.10] = 0

        if deficit.sum() > 0:

            weights = deficit**2

            yc = (
                np.sum(y_D * weights)
                / np.sum(weights)
            )

        else:
            yc = np.nan

        centerline_y.append(yc)

    centerline_y = np.array(centerline_y)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    plt.figure(figsize=(12, 4))

    # Turbines
    for x_turb in [0, 7, 14, 21]:
        plt.plot(
            [x_turb, x_turb],
            [-0.5, 0.5],
            color='0.6',      # grey
            linewidth=4
        )

    # LES wake centerline
    plt.plot(
        x_D,
        centerline_y,
        color='black',
        linewidth=3,
        label=r'$LES_{t_{avg}}$'
    )

    plt.xlabel("x/D")
    plt.ylabel("y/D")

    plt.xlim(x_D.min(), x_D.max())
    plt.ylim(-5.0, 5.0)

    plt.grid(True, alpha=0.3)

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def compute_wake_centerline(U, y_D):
    """
    U shape = (ny, nx)
    """

    centerline_y = []

    for j in range(U.shape[1]):

        profile = U[:, j]

        deficit = np.maximum(1 - profile, 0)

        deficit[deficit < 0] = 0

        if deficit.sum() > 0:

            weights = deficit**2

            yc = np.sum(y_D * weights) / np.sum(weights)

        else:
            yc = np.nan

        centerline_y.append(yc)

    return np.array(centerline_y)

def compare_rans_les_centerline():

    # ===== LES =====
    df = pd.read_csv(
        r"LES\extracted_statistics\Results\60_min_average\mean_wake_t0_3600_long.csv"
    )

    heatmap = df.pivot(
        index="y",
        columns="x",
        values="U_Uref"
    )

    x_les = heatmap.columns.values
    y_les = heatmap.index.values

    centerline_les = compute_wake_centerline(
        heatmap.values,
        y_les
    )

    # ===== RANS =====
    ds_rans = xr.open_dataset(
        r"RANS\data\flowdata_basecase10.nc"
    )

    U_rans = (
        ds_rans["U"]
        .interp(z=170.0)
        .values.T
    )

    x_rans = ds_rans.x.values / 284.0
    y_rans = ds_rans.y.values / 284.0

    # Normalize
    U_rans = U_rans / np.max(U_rans)

    centerline_rans = compute_wake_centerline(
        U_rans,
        y_rans
    )

    # ===== Plot =====
    plt.figure(figsize=(12,4))

    for x_turb in [0, 7, 14, 21]:
        plt.plot(
            [x_turb, x_turb],
            [-0.5, 0.5],
            color="0.5",
            linewidth=4
        )

    plt.plot(
        x_les,
        centerline_les,
        "k",
        lw=3,
        label="LES",
        color="black"
    )

    plt.plot(
        x_rans,
        centerline_rans,
        "--",
        lw=3,
        label="RANS",
        color="black"
    )

    plt.xlabel("x/D")
    plt.ylabel("y/D")
    plt.xlim(-4, 32)
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()

def rans_ta_les_mean_velocity_difference():

    df = pd.read_csv(
        r"LES\extracted_statistics\Results\60_min_average\mean_wake_t0_3600_long.csv"
    )

    print(df.columns)

    heatmap = df.pivot(
        index="y",
        columns="x",
        values="U_Uref"
    )

    x_les = heatmap.columns.values
    y_les = heatmap.index.values

    U_les = heatmap.values

    # ==========================================
    # RANS
    # ==========================================
    ds_rans = xr.open_dataset(
        r"RANS\data\flowdata_basecase10.nc"
    )

    U_rans = (
        ds_rans["U"]
        .interp(z=170.0)
        .values.T
    )

    x_rans = ds_rans.x.values / 284.0
    y_rans = ds_rans.y.values / 284.0

    # normalize
    U_rans = U_rans / np.max(U_rans)

    # ==========================================
    # Interpolate RANS -> LES grid
    # ==========================================
    interp = RegularGridInterpolator(
        (y_rans, x_rans),
        U_rans,
        bounds_error=False,
        fill_value=np.nan
    )

    X_les, Y_les = np.meshgrid(
        x_les,
        y_les
    )

    points = np.column_stack([
        Y_les.ravel(),
        X_les.ravel()
    ])

    U_rans_interp = interp(points).reshape(
        U_les.shape
    )

    # ==========================================
    # Difference
    # ==========================================
    difference = U_rans_interp - U_les

    # ==========================================
    # Plot
    # ==========================================
    plt.figure(figsize=(12,4))

    c = plt.contourf(
        x_les,
        y_les,
        difference,
        levels=np.linspace(-0.15, 0.15, 100),
        cmap="RdBu_r",
        extend="both"
    )

    plt.colorbar(
        c,
        label=r"$U_{RANS}/U_{ref}-U_{LES}/U_{ref}$"
    )

    for x_turb in [0,7,14,21]:
        plt.plot(
            [x_turb,x_turb],
            [-0.5,0.5],
            "k",
            lw=3
        )

    plt.xlabel("x/D")
    plt.ylabel("y/D")

    plt.title(
        "RANS - LES Mean Velocity Difference"
    )

    plt.tight_layout()
    plt.show()

def plot_rans_mean_wake():

    data = xr.open_dataset(
        r"RANS\data\flowdata_basecase10.nc"
    )

    D = 284.0

    x = data.x.values
    y = data.y.values

    X, Y = np.meshgrid(
        x / D,
        y / D
    )

    U = (
        data["U"]
        .interp(z=170.0, method="linear")
        .values
    )

    U = U / 9.5

    plt.figure(figsize=(12, 4))

    pcm = plt.pcolormesh(
        X,
        Y,
        U.T,              # IMPORTANT
        cmap="plasma",
        shading="auto",
        vmin=0.0,
        vmax=1.1
    )

    plt.colorbar(
        pcm,
        label=r"$U/U_{ref}$"
    )

    for x_turb in [0, 7, 14, 21]:
        plt.plot(
            [x_turb, x_turb],
            [-0.5, 0.5],
            color="black",
            linewidth=4
        )

    plt.xlabel(r"$x/D$")
    plt.ylabel(r"$y/D$")

    plt.xlim(-3, 32)
    plt.ylim(-5, 5)

    plt.title("RANS Mean Wake")

    plt.tight_layout()
    plt.show()

def compute_ct(csv_file, x_turb):
    """
    Compute CT from a full wake field.

    Parameters
    ----------
    csv_file : str
        LES wake field csv

    x_turb : float
        Turbine location in x/D
        e.g. 0, 7, 14, 21

    Returns
    -------
    ct : float
    Ud_Uinf : float
    """

    df = pd.read_csv(csv_file)

    # Profile nearest turbine
    x_actual = min(df["x"].unique(),
                   key=lambda x: abs(x - x_turb))

    profile = df[df["x"] == x_actual].copy()

    # Rotor region
    mask = np.abs(profile["y"]) <= 0.5

    Ud_Uinf = profile.loc[mask, "U_Uref"].mean()

    a = 1 - Ud_Uinf

    ct = 4 * a * (1 - a)

    return ct

