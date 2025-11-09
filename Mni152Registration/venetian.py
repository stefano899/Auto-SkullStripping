# -*- coding: utf-8 -*-
# Light smoothing con ANTs (antspyx)
# - legge un NIfTI
# - applica uno smoothing gaussiano leggero
# - salva il risultato

import ants

# ====== CONFIG ======
IN_PATH  = r"C:\Users\Stefano\Desktop\Stefano\Datasets\DsTrCycleRegisteredMseg\Testing\T1_Brain\fake\Testing_Center_01_Patient_10_FLAIR_preprocessed_inMNI_resampled_fake_EPOCH_220.nii.gz"
OUT_PATH = r"C:\Users\Stefano\Desktop\Stefano\Datasets\DsTrCycleRegisteredMseg\Testing\T1_Brain\fake\Testing_Center_01_Patient_10_FLAIR_preprocessed_inMNI_resampled_fake_EPOCH_220-smooth.nii.gz"

MODE = "iso"         # "iso" = isotropico; "z" = solo lungo z
SIGMA_MM = 0.6    # intensità di smoothing in millimetri (0.6–1.0 mm di solito è o0.8
KEEP_BG_ZERO = True  # mantieni lo sfondo (<=0) a zero dopo lo smoothing
# ====================

# Leggi il volume (preserva origin/spacing/direction)
img = ants.image_read(IN_PATH)

# Scegli il vettore sigma in mm
if MODE.lower() == "iso":
    sigma = SIGMA_MM
elif MODE.lower() == "z":
    # smoothing solo lungo z: usa ~0 mm in x,y (un epsilon) e sigma in z
    sigma = [1e-6, 1e-6, SIGMA_MM]
else:
    raise ValueError("MODE deve essere 'iso' o 'z'")

# Applica smoothing in unità fisiche (mm)
sm = ants.smooth_image(img, sigma=sigma, sigma_in_physical_coordinates=True)

# (Opzionale) ripristina lo sfondo a zero
if KEEP_BG_ZERO:
    mask_bg = img <= 0
    sm = sm * (mask_bg == 0)

# Salva
ants.image_write(sm, OUT_PATH)
print("Smoothing fatto →", OUT_PATH)
print("Spacing (mm):", img.spacing, "  Mode:", MODE, "  Sigma(mm):", sigma)
