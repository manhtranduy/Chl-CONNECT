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

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(lon, lat, chl, shading="auto", cmap="viridis")
    plt.colorbar(label="Chl-a (mg m$^{-3}$)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("CONNECT Chlorophyll-a")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
