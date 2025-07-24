from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt

from common.Chl_CONNECT import Chl_CONNECT


def main():
    file_path = "tests/S2B_MSIL1C_20240718T133839_N0510_R124_T22MHE_20240718T151854.SAFE_L2_OCSMART.h5"
    with h5py.File(file_path, "r") as f:
        lon = f["Longitude"][:]
        lat = f["Latitude"][:]
        rrs = np.stack([
            f["Rrs"]["Rrs_442nm"][:],
            f["Rrs"]["Rrs_492nm"][:],
            f["Rrs"]["Rrs_559nm"][:],
            f["Rrs"]["Rrs_665nm"][:],
            f["Rrs"]["Rrs_704nm"][:],
        ], axis=-1)

    connect = Chl_CONNECT(rrs, sensor="MSI", block_size=(256, 256))
    chl = connect.Chl_comb
    classes = connect.Class

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im = axes[0].pcolormesh(lon, lat, chl, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="Chl-a (mg m$^{-3}$)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("CONNECT Chlorophyll-a")

    class_im = axes[1].pcolormesh(lon, lat, classes, shading="auto", cmap="tab10", vmin=1, vmax=5)
    fig.colorbar(class_im, ax=axes[1], label="OWT class")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("OWT Classification")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
