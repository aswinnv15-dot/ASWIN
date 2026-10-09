import numpy as np
import matplotlib
matplotlib.use("Agg")          # remove this line in PyCharm if you want the plot window
import matplotlib.pyplot as plt

# ---------------- inputs (your original values) ----------------
Ta = 30             # surrounding temperature, deg C
Ts = 800            # initial temperature, deg C
h = 30              # convective heat transfer coefficient, W/m2.K
rho = 7600          # density, kg/m3
k = 45              # conductivity, W/m.K (only used for Biot number)
cp = 500            # specific heat, J/kg.K
side = 0.005        # cube side, m
dt = 0.1            # time step, s (smaller than 1 s: the pearlite nose is ~1 s)
TotalT = 600        # total time, s

# ---------------- eutectoid steel (~Fe-0.8C) transformation data ----------------
Ae1 = 727.0         # eutectoid temperature, deg C
Ms = 220.0          # martensite start, deg C
km = 0.011          # Koistinen-Marburger constant, 1/K

# Approximate isothermal TTT data (1% start, 99% finish times in s) read from a
# typical 1080 steel diagram. Replace with your own diagram values if you have them.
# columns: T(C), t_start, t_finish
pearlite = np.array([
    [700, 100, 1500],
    [650,   8,   60],
    [600,   2,   12],
    [550,   1,    6],     # nose
    [500,   3,   20],
    [450,  10,   80],
])
bainite = np.array([
    [450,  20,  300],
    [400,   6,  100],
    [350,  15,  250],
    [300,  40,  600],
    [250, 150, 2500],
])

def make_avrami(table):
    """X = 1 - exp(-b t^n) fitted so X = 1% at t_start and 99% at t_finish."""
    T = table[::-1, 0]                              # ascending T for np.interp
    ts, tf = table[::-1, 1], table[::-1, 2]
    n = np.log(np.log(1 / 0.01) / np.log(1 / 0.99)) / np.log(tf / ts)
    b = -np.log(0.99) / ts ** n
    return T, n, b

def avrami_params(model, Tc):
    T, n, b = model
    if Tc < T[0] or Tc > T[-1]:
        return None                                 # outside the C-curve range
    return np.interp(Tc, T, n), np.exp(np.interp(Tc, T, np.log(b)))

P_model = make_avrami(pearlite)
B_model = make_avrami(bainite)

# ---------------- setup ----------------
v = side ** 3               # volume, m3
A = 4 * side ** 2           # only 4 faces exposed, m2
m = rho * v                 # mass, kg

n_steps = int(round(TotalT / dt))
Q = np.zeros(n_steps)       # heat lost in each step, J
Temp = np.zeros(n_steps)    # temperature after each step
time = np.zeros(n_steps)    # time at end of each step
fP = np.zeros(n_steps)      # pearlite fraction
fB = np.zeros(n_steps)      # bainite fraction
fM = np.zeros(n_steps)      # martensite fraction

Bi = h * (v / A) / k
print(f"Biot number = {Bi:.5f}  (lumped model is valid for Bi < 0.1)")

# ---------------- heat balance + phase transformation loop ----------------
xP = xB = xM = 0.0          # actual volume fractions
yP = yB = 0.0               # Avrami (extended) variables used for additivity

def advance(model, Tc, y_old):
    """Scheil additivity: find the fictive time for y_old at this T, advance by dt."""
    p = avrami_params(model, Tc)
    if p is None:
        return y_old
    n, b = p
    t_fict = (-np.log(1 - y_old) / b) ** (1 / n) if y_old > 0 else 0.0
    return 1 - np.exp(-b * (t_fict + dt) ** n)

for i in range(n_steps):
    Q[i] = h * A * (Ts - Ta) * dt       # q = h*A*(Ts - Ta), times dt
    dT = Q[i] / (m * cp)                # Q = m*cp*dT
    Ts = Ts - dT
    Temp[i] = Ts
    time[i] = (i + 1) * dt

    austenite = 1.0 - xP - xB - xM
    if austenite > 1e-9:
        if Ts < Ae1:
            # diffusional products compete for the remaining austenite
            if Ts > Ms:
                yP_new = min(advance(P_model, Ts, yP), 1 - 1e-12)
                yB_new = min(advance(B_model, Ts, yB), 1 - 1e-12)
                dP = austenite * (yP_new - yP) / (1 - yP)
                dB = (austenite - dP) * (yB_new - yB) / (1 - yB)
                yP, yB = yP_new, yB_new
                xP += dP
                xB += dB
            # athermal martensite from the austenite that is still left
            if Ts < Ms:
                austenite = 1.0 - xP - xB - xM
                x_target = (1.0 - xP - xB) * (1 - np.exp(-km * (Ms - Ts)))
                xM = max(xM, x_target)
    fP[i], fB[i], fM[i] = xP, xB, xM

fA = 1.0 - fP - fB - fM
print(f"Temperature at {TotalT} s = {Temp[-1]:.2f} C")
print(f"Final phases: pearlite {fP[-1]:.1%}, bainite {fB[-1]:.1%}, "
      f"martensite {fM[-1]:.1%}, retained austenite {fA[-1]:.1%}")

step = int(round(60 / dt))
for i in range(step - 1, n_steps, step):
    print(f"t = {time[i]:6.1f} s   T = {Temp[i]:8.2f} C   "
          f"P = {fP[i]:.3f}  B = {fB[i]:.3f}  M = {fM[i]:.3f}")

# ---------------- plot ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
ax1.plot(time, Temp)
ax1.axhline(Ae1, ls="--", c="gray", lw=0.8)
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("Cooling curve")
ax1.grid(True)

ax2.plot(time, fP, label="Pearlite")
ax2.plot(time, fB, label="Bainite")
ax2.plot(time, fM, label="Martensite")
ax2.plot(time, fA, label="Austenite", ls="--")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Volume fraction")
ax2.set_title("Phase fractions")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig("cooling_phases.png", dpi=120)
plt.show()
