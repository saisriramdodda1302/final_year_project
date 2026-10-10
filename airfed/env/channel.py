"""Radio and power formulas from Section III of the paper.

Everything is vectorised with numpy so one call handles all UAV/device pairs.
"""

import numpy as np


def db_to_linear(db):
    return 10.0 ** (np.asarray(db) / 10.0)


def dbm_to_watt(dbm):
    return 10.0 ** ((np.asarray(dbm) - 30.0) / 10.0)


def watt_to_dbm(w):
    return 10.0 * np.log10(np.maximum(w, 1e-30)) + 30.0


def ground_distance(uav_pos, dev_pos):
    """3-D distance from every UAV to every device (Eq. 2). Returns K x M."""
    dx = uav_pos[:, None, 0] - dev_pos[None, :, 0]
    dy = uav_pos[:, None, 1] - dev_pos[None, :, 1]
    return np.sqrt(dx**2 + dy**2 + uav_pos[:, None, 2] ** 2)


def uav_distance(uav_pos):
    """Distance between every pair of UAVs. Returns K x K."""
    diff = uav_pos[:, None, :] - uav_pos[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def ground_gain(cfg, dist):
    """|h|^2 for air-to-ground links (Eq. 3), with our path-loss exponent."""
    return db_to_linear(cfg.g0_db) * dist ** (-cfg.ground_exponent)


def air_gain(cfg, dist):
    """|h|^2 between two UAVs (Eq. 14): line of sight, exponent 2."""
    return db_to_linear(cfg.g_inter_db) / np.maximum(dist, 1.0) ** 2


def ground_rssi_dbm(cfg, dist):
    """Signal a device sees from each UAV (Eq. 4), in dBm."""
    return watt_to_dbm(cfg.p_uav_tx * ground_gain(cfg, dist))


def air_rssi_dbm(cfg, dist):
    return watt_to_dbm(cfg.p_uav_tx * air_gain(cfg, dist))


def uplink_rates(cfg, gain, serving, airtime):
    """Shannon uplink rate for each device to its serving UAV (Eq. 9).

    gain:    K x M channel gains
    serving: serving UAV per device (-1 if uncovered)
    airtime: fraction of the step each device spends transmitting.

    Devices in the same cell take turns (the UAV schedules them), so only
    devices attached to *other* UAVs interfere, and only for the share of
    the step they are actually sending.
    """
    noise = dbm_to_watt(cfg.noise_dbm)
    rates = np.zeros(len(serving))
    for m, k in enumerate(serving):
        if k < 0:
            continue
        received = cfg.p_dev_tx * gain[k]          # power from every device at UAV k
        other_cells = (serving != k) & (serving >= 0)
        interference = (received * airtime * other_cells).sum()
        rates[m] = cfg.bw * np.log2(1.0 + received[m] / (noise + interference))
    return rates


def downlink_rates(cfg, gain, serving):
    """UAV -> device rate. The paper says 'similar to uplink with higher power'."""
    noise = dbm_to_watt(cfg.noise_dbm)
    rates = np.zeros(len(serving))
    ok = serving >= 0
    g = gain[serving[ok], np.nonzero(ok)[0]]
    rates[ok] = cfg.bw * np.log2(1.0 + cfg.p_uav_tx * g / noise)
    return rates


def air_rates(cfg, dist):
    """UAV <-> UAV rate (Eq. 13). Returns K x K, zero on the diagonal."""
    noise = dbm_to_watt(cfg.noise_dbm)
    rate = cfg.bw_inter * np.log2(1.0 + cfg.p_uav_tx * air_gain(cfg, dist) / noise)
    np.fill_diagonal(rate, 0.0)
    return rate


def flight_power(cfg, speed):
    """Hover power plus parasitic drag (Eq. 19)."""
    return cfg.p_hover + 0.5 * cfg.rho_air * cfg.drag_area * cfg.drag_coef * speed**3
