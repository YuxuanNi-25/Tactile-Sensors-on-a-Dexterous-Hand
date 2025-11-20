#!/usr/bin/env python3
"""
Load 'wrench_log.npy' produced by sim_tactile.py and plot fingertip
force magnitudes (in the palm frame) over time.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.load("wrench_log.npy", allow_pickle=True)

    num_frames = len(data)
    print("num frames:", num_frames)

    # Auto-detect link indices from the first frame
    first_keys = list(data[0].keys())
    first_keys = sorted(first_keys)
    print("fingertip link indices:", first_keys)

    # Map link indices to finger names
    finger_names = ["thumb", "index", "middle", "ring", "little"]
    label_map = {}

    for i, link_idx in enumerate(first_keys):
        if i < len(finger_names):
            label_map[link_idx] = finger_names[i]
        else:
            label_map[link_idx] = f"link {link_idx}"

    t = np.arange(num_frames)

    plt.figure()
    for link_idx in first_keys:
        F_mag = []
        for frame in data:
            wrench = frame[link_idx]
            # Use the force in palm frame (common reference frame)
            F = wrench["F_palm"]
            F_mag.append(np.linalg.norm(F))
        F_mag = np.array(F_mag)
        plt.plot(t, F_mag, label=label_map[link_idx])

    plt.xlabel("time step")
    plt.ylabel("force magnitude in palm frame (N)")
    plt.title("Fingertip force magnitude while closing against a rigid plate")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
