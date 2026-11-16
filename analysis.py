"""Class for wake simulation and visualization."""
import matplotlib.pyplot as plt

class WakeSimulation:
    def __init__(self, ds, D, U_ref):
        self.ds = ds
        self.D = D
        self.U_ref = U_ref


    def plot_instant_wake(self, t):
        """Method for plotting the instantaneous wake at a given time step."""
        
        # Create normalized coordinates
        U = self.ds["U"].isel(t=t, z=0) / self.U_ref
        U_norm = U.assign_coords(
            x=U.x / self.D,
            y=U.y / self.D
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
        plt.title(f"U/Uref at t={float(self.ds.t[t]):.1f} s")
        plt.show()