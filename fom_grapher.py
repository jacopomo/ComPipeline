import matplotlib.pyplot as plt
import numpy as np
 
E = [100,125,160,200,250,315,500,1000,1600,3200,5000,16000,25000]
n1 = [0,0.005,0.1,0.192,0.177,0.234,0.411,0.542,0.58,0.645,0.664,0.59,0.6]
n2 = [0,0.016,0.038,0.028,0.027,0.031,0.064,0.29,0.426,0.578,0.623,0.583,0.597]
c3 = [0, 0.003,0.08,0.182,0.305,0.418,0.498,0.532,0.538,0.551,0.564,0.478,0.472]

###

f = np.linspace(0,1,100)
acd = 0.648*0.686*f + (1-0.0643)*(1-f)
rf = 0.796*0.328*f + (1-0.0103)*(1-f)
pca = 0.788*0.212*f + (1-0.00418)*(1-f)



series = {
    r'$2$-class $n_{hits}>1$': n1,
    r'$2$-class $n_{hits}>2$': n2,
    r'$3$-class $n_{hits}>1$': c3,
}
colors = ['C0', 'C1', 'C2']
markers = ['o', 's', 'v']
 
fig, ax = plt.subplots(figsize=(8, 6))
 
# Scatter points + dotted connecting lines for each series

for (label, y), color, marker in zip(series.items(), colors, markers):
    ax.plot(E, y, linestyle=':', marker=marker, color=color, label=label)
 
# --- Shade the regions where each model is the best (highest) ---
# Winner at each E point, then shade each interval by whichever model
# is on top there (using the midpoint of the interval as the check).
values = np.array(list(series.values()))  # shape (n_series, n_points)
E_arr = np.array(E, dtype=float)


for i in range(len(E) - 1):
    # interpolate each series at the interval midpoint (in log-x space)
    mid = np.sqrt(E_arr[i] * E_arr[i + 1])
    mid_vals = [
        np.interp(mid, E_arr[i:i + 2], values[j, i:i + 2])
        for j in range(values.shape[0])
    ]
    winner = int(np.argmax(mid_vals))
    ax.axvspan(E[i], E[i + 1], color=colors[winner], alpha=0.12, lw=0)
 
ax.set_xscale('log')
ax.set_yscale('log')
 
ax.set_xlabel('Energy [keV]')
ax.set_ylabel('P(TP)')
ax.set_title('Figure of Merit (Probability of a "True Positive") at Various Energy Bins')
ax.grid(True, which='both', linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.savefig('fom_vs_E.png', dpi=150)
plt.show()

seriesf = {
    r'ACD only': acd,
    r'Random Forest': rf,
    r'PCA': pca,
}

fig, ax = plt.subplots(figsize=(8, 6))
for (label, y), color, marker in zip(seriesf.items(), colors, markers):
    ax.plot(f, y, linestyle='-', color=color, label=label)
values = np.array(list(seriesf.values()))  # shape (n_series, n_points)
f_arr = np.array(f, dtype=float)


for i in range(len(f) - 1):
    # interpolate each series at the interval midpoint (in log-x space)
    mid = np.sqrt(f_arr[i] * f_arr[i + 1])
    mid_vals = [
        np.interp(mid, f_arr[i:i + 2], values[j, i:i + 2])
        for j in range(values.shape[0])
    ]
    winner = int(np.argmax(mid_vals))
    ax.axvspan(f[i], f[i + 1], color=colors[winner], alpha=0.12, lw=0)

    
ax.set_xlabel(r'Fraction of signal $f$')
ax.set_ylabel('Figure of Merit')
ax.set_title('Figure of Merit (Probability of a "Correct Classification") vs Fraction of Signal')
ax.grid(True, which='both', linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.savefig('fom_vs_f.png', dpi=150)
plt.show()