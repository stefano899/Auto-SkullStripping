import nibabel as nib
from nibabel.processing import resample_to_output
from pathlib import Path
import numpy as np

# === PATH ===
in_path = Path(r"C:\Users\stefa\Documents\cervelli\sub-032302\ses-01\anat\sub-032302_ses-01_acq-lowres_FLAIR.nii.gz")
out_dir = Path(r"C:\Users\stefa\Documents\cervelli\sub-032302\ses-01\anat\preproc_out")

# 🔧 CREA LA CARTELLA SE NON C'È
out_dir.mkdir(parents=True, exist_ok=True)

# usa lo stesso id (032302) anche nel nome di output
out_path_resampled = out_dir / "sub-032302_ses-01_acq-lowres_FLAIR_1mm.nii.gz"
out_path_clean = out_dir / "sub-032302_ses-01_acq-lowres_FLAIR_1mm_zclean.nii.gz"

print(f"[INFO] Carico: {in_path}")
img = nib.load(str(in_path))

data = img.get_fdata()
affine = img.affine
hdr = img.header

print("\n=== INFO ORIGINALE ===")
print(f"shape: {img.shape}")
print(f"dtype (dati caricati): {data.dtype}")
print("affine:")
print(affine)

zooms = hdr.get_zooms()[:3]
dx, dy, dz = zooms
print(f"zoom header (mm): X={dx}, Y={dy}, Z={dz}")

coord0 = affine @ np.array([0, 0, 0, 1])
coord1 = affine @ np.array([0, 0, 1, 1])
print(f"distanza reale tra slice consecutive (da affine): {np.linalg.norm(coord1[:3]-coord0[:3]):.3f} mm")

# === RICAMPIONAMENTO ===
print("\n[INFO] Ricampiono a (1.0, 1.0, 1.0) mm")
resampled = resample_to_output(img, voxel_sizes=(1.0, 1.0, 1.0), order=0)

resampled_data = resampled.get_fdata(dtype=np.float32)
new_affine = resampled.affine
new_hdr = resampled.header.copy()
new_hdr.set_data_dtype(np.float32)
new_hdr.set_zooms((1.0, 1.0, 1.0))

resampled_img = nib.Nifti1Image(resampled_data, new_affine, header=new_hdr)
resampled_img.set_qform(new_affine, code=1)
resampled_img.set_sform(new_affine, code=1)

print(f"\n[INFO] Salvo ricampionato in: {out_path_resampled}")
nib.save(resampled_img, str(out_path_resampled))

# --- opzionale: z-score e pulizia ---
mean_val = resampled_data.mean()
std_val = resampled_data.std()
zvol = (resampled_data - mean_val) / std_val
zvol_clean = np.where(zvol > 0, zvol, 0).astype(np.float32)

clean_img = nib.Nifti1Image(zvol_clean, new_affine, header=new_hdr)
clean_img.set_qform(new_affine, code=1)
clean_img.set_sform(new_affine, code=1)

print(f"[INFO] Salvo volume pulito in: {out_path_clean}")
nib.save(clean_img, str(out_path_clean))

print("\n=== FATTO ===")
