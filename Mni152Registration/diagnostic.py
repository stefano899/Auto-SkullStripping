# Requisiti (se servono):
# pip install nibabel numpy matplotlib

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import csv, os

# ====== CONFIGURA QUI ======
NIFTI_PATH = r"C:\Users\Stefano\Desktop\Stefano\Datasets\DsTrCycleRegisteredMseg\Testing\Flair_Brain\volume_flair_111_testing_center01_patient10_inMNI-venetian_bias.nii.gz"
AXIS = 2                       # 0=x, 1=y, 2=z (tipico z)
MASK_AUTO = True               # True = usa maschera foreground per-slice (percentile)
SAVE_CSV = True                # salva anche un CSV con i valori per slice
OUTDIR = "./out"               # dove salvare PNG/CSV
PCT_THRESHOLD = 30.0           # percentile per la maschera (se MASK_AUTO=True)
# ===========================

os.makedirs(OUTDIR, exist_ok=True)

# Carica NIfTI
img = nib.load(NIFTI_PATH)
data = img.get_fdata(caching='unchanged')
hdr = img.header
zooms = hdr.get_zooms()[:3] if len(hdr.get_zooms()) >= 3 else (1.0,1.0,1.0)

print("=== Header ===")
print("Shape:", data.shape)
print("Voxel size (mm):", zooms)
try:
    print("Datatype:", hdr.get_data_dtype())
except Exception:
    print("Datatype:", data.dtype)

# Funzione per estrarre una slice lungo AXIS
def get_slice(vol, axis, i):
    return vol[i,:,:] if axis==0 else (vol[:,i,:] if axis==1 else vol[:,:,i])

# Calcola mean/median per slice (con o senza maschera foreground)
n_slices = data.shape[AXIS]
means = np.zeros(n_slices, dtype=float)
medians = np.zeros(n_slices, dtype=float)
counts = np.zeros(n_slices, dtype=int)

for i in range(n_slices):
    sl = get_slice(data, AXIS, i)
    if MASK_AUTO:
        thr = np.percentile(sl, PCT_THRESHOLD)
        msk = sl > thr
        vals = sl[msk] if np.count_nonzero(msk) >= sl.size*0.05 else sl.reshape(-1)
    else:
        vals = sl.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        means[i] = np.nan
        medians[i] = np.nan
        counts[i] = 0
    else:
        means[i] = float(np.mean(vals))
        medians[i] = float(np.median(vals))
        counts[i] = int(vals.size)

# Metriche diagnostiche
diffs = np.abs(np.diff(means))
mean_diff = float(np.nanmean(diffs))
p95_diff  = float(np.nanpercentile(diffs, 95))

even = means[0::2]
odd  = means[1::2]
if even.size and odd.size:
    m_even = float(np.nanmean(even))
    m_odd  = float(np.nanmean(odd))
    odd_even_ratio = abs(m_even - m_odd) / (abs((m_even + m_odd)/2.0) + 1e-8)
else:
    odd_even_ratio = np.nan

print("\n=== Slice-wise intensity diagnostics (axis=%s) ===" % {0:'x',1:'y',2:'z'}[AXIS])
print("Numero di slice:", n_slices)
print("Media |Δ| tra slice consecutive:", mean_diff)
print("95° percentile |Δ|:", p95_diff)
print("Odd/Even mean difference ratio:", odd_even_ratio)
if np.isfinite(odd_even_ratio) and odd_even_ratio > 0.03:
    print(">> Offset pari/dispari evidente (possibile interleave).")
if np.isfinite(p95_diff) and p95_diff > 0.05*np.nanmedian(means):
    print(">> Variazione forte tra slice (artefatto d'intensità probabile).")

# Plot profilo
png_path = os.path.join(OUTDIR, f"slice_intensity_profile_axis-{AXIS}.png")
plt.figure()
plt.plot(means, label="mean" + (" (masked)" if MASK_AUTO else " (all)"))
plt.plot(medians, linestyle='--', label="median")
plt.title(f"Per-slice intensity profile (axis={['x','y','z'][AXIS]})")
plt.xlabel("Slice index")
plt.ylabel("Intensity")
plt.legend()
plt.tight_layout()
plt.savefig(png_path, dpi=160)
print("Salvato plot:", png_path)

# (Opzionale) CSV
if SAVE_CSV:
    csv_path = os.path.join(OUTDIR, f"slice_intensity_profile_axis-{AXIS}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice_index","mean","median","count"])
        for i,(m1,m2,c) in enumerate(zip(means, medians, counts)):
            w.writerow([i, float(m1), float(m2), int(c)])
    print("Salvato CSV:", csv_path)
