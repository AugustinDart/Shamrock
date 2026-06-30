"""
Binary orbit functions
=======================================
"""

import numpy as np
import matplotlib.pyplot as plt
import shamrock as chama

chama.matplotlib.set_shamrock_mpl_style()

# =========================================================
# UNITES
# =========================================================

si = chama.UnitSystem()
sicte = chama.Constants(si)

codeu = chama.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)

ucte = chama.Constants(codeu)
G = ucte.G()

# =========================================================
# PARAMETRES
# =========================================================

T = 10
dt = 0.001
n_steps = int(T / dt)

# =========================================================
# ROTATION
# =========================================================

def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])

    return Rz @ Ry @ Rx

# =========================================================
# CONDITIONS INITIALES
# =========================================================

def binary_initial_conditions(m1, m2, a, e, nu=0.0,
                              G=G, roll=0.0, pitch=0.0, yaw=0.0):

    M = m1 + m2
    r = a * (1 - e**2) / (1 + e*np.cos(nu))

    h = np.sqrt(G*M*a*(1 - e**2))
    vr = G*M/h * e*np.sin(nu)
    vtheta = h / r

    x_rel = np.array([r*np.cos(nu), r*np.sin(nu), 0])
    v_rel = np.array([
        vr*np.cos(nu) - vtheta*np.sin(nu),
        vr*np.sin(nu) + vtheta*np.cos(nu),
        0
    ])

    x1 = -m2/M * x_rel
    x2 =  m1/M * x_rel
    v1 = -m2/M * v_rel
    v2 =  m1/M * v_rel

    if roll or pitch or yaw:
        R = rotation_matrix(roll,pitch,yaw)
        x1, x2 = R@x1, R@x2
        v1, v2 = R@v1, R@v2

    return x1, x2, v1, v2

# =========================================================
# MODEL
# =========================================================

def build_binary_sph_model(m1,m2,a,e,racc=0.001):

    ctx = chama.Context()
    ctx.pdata_layout_new()

    model = chama.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    chama.enable_experimental_features()

    cfg = model.gen_default_config()
    cfg.set_self_gravity_none()
    cfg.set_particle_mass(1e-6)
    cfg.set_eta_sink(0.01)
    cfg.set_units(codeu)

    model.set_solver_config(cfg)

    # Conditions iniatiales identiques à l'autre code pour la comparaison
    r0 = 1.0
    v0 = np.sqrt(1 - e) * np.sqrt(G * (m1 + m2) / r0)

    x_rel = np.array([r0, 0.0, 0.0])
    v_rel = np.array([0.0, v0, 0.0])

    M = m1 + m2

    x1 = (m2 / M) * x_rel
    x2 = -(m1 / M) * x_rel

    v1 = (m2 / M) * v_rel
    v2 = -(m1 / M) * v_rel

    model.add_sink(m1, tuple(x1), tuple(v1), racc)
    model.add_sink(m2, tuple(x2), tuple(v2), racc)

    model.init_scheduler(10_000_000, 1)

    ext = max(1.0, a*5)
    model.resize_simulation_box((-ext,-ext,-ext),(ext,ext,ext))

    model.set_dt(dt)

    return ctx, model

# =========================================================
# RUN
# =========================================================

def run_binary(model):

    snapshots = []
    t = 0.0

    # =====================================================
    # Sauvegarde des conditions initiales (t = 0)
    # =====================================================
    pos, vel = get_sinks(model)

    snapshots.append({
        "time": t,
        "positions": pos,
        "velocities": vel
    })

    # =====================================================
    # Évolution de la simulation
    # =====================================================
    for _ in range(n_steps):

        model.evolve_once_override_time(t, dt)
        t += dt

        pos, vel = get_sinks(model)

        snapshots.append({
            "time": t,
            "positions": pos,
            "velocities": vel
        })

    return snapshots


def get_sinks(model):
    sinks = model.get_sinks()
    pos = [tuple(s["pos"]) for s in sinks]
    vel = [tuple(s["velocity"]) for s in sinks]
    return pos, vel

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    m1, m2 = 1, 0.000006
    a, e = 1.0, 0.7

    ctx, model = build_binary_sph_model(m1,m2,a,e)
    snapshots = run_binary(model)

    # =========================================================
    # 🔥 EXPORT TOUS LES 0.25 ANS (IMPORTANT)
    # =========================================================

    dt_out = 0.25

    times = np.array([s["time"] for s in snapshots])

    # indices correspondant à 0.25 ans
    idx_out = np.arange(0, len(snapshots), int(dt_out/dt))

    snaps = [snapshots[i] for i in idx_out]

    times = np.array([s["time"] for s in snaps])

    pos1 = np.array([s["positions"][0] for s in snaps])
    pos2 = np.array([s["positions"][1] for s in snaps])

    vel1 = np.array([s["velocities"][0] for s in snaps])
    vel2 = np.array([s["velocities"][1] for s in snaps])

    data = np.column_stack([
        times,

        pos1[:,0], pos1[:,1], pos1[:,2],
        vel1[:,0], vel1[:,1], vel1[:,2],

        pos2[:,0], pos2[:,1], pos2[:,2],
        vel2[:,0], vel2[:,1], vel2[:,2],
    ])

    header = (
        "t,"
        "x1,y1,z1,vx1,vy1,vz1,"
        "x2,y2,z2,vx2,vy2,vz2"
    )

    np.savetxt(
        "binary_shamrock_025yr.csv",
        data,
        delimiter=",",
        header=header,
        comments=""
    )

    print("✔ Export 0.25 ans OK -> binary_shamrock_025yr.csv")