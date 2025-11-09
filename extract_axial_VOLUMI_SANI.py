import os
import re
import shutil
import nibabel as nib
import numpy as np
from skimage.transform import resize
from nibabel.processing import resample_to_output
import imageio
from pathlib import Path

# ========================================
# CONFIG
# ========================================
tipo = "VOLUMISANI"
base_root = r"E:\Datasets\Volumi_sani_1mm_MNI"

# 🔹 La cartella train è fuori da VOLUMI-SANI-1mm
DATASETS_ROOT = r"E:\Datasets"
GLOBAL_TRAIN_A = os.path.join(DATASETS_ROOT, "train", "trainA")  # T1
GLOBAL_TRAIN_B = os.path.join(DATASETS_ROOT, "train", "trainB")  # FLAIR
os.makedirs(GLOBAL_TRAIN_A, exist_ok=True)
os.makedirs(GLOBAL_TRAIN_B, exist_ok=True)

# modalità di riferimento per il crop
ref_modality = "FLAIR"

# parametri
nz_threshold = 5000
resize_shape = (256, 256)
voxel_sizes = (1, 1, 1)


# ========================================
# DISCOVERY DELLE CARTELLE
# ========================================
def discover_subject_anat_pairs(root: str):
    """
    Ritorna tutte le coppie (subj, path_relativo) dove esiste anat/skullstripped.
    """
    root_p = Path(root)
    # ordiniamo così prendiamo davvero "il primo" sempre nello stesso modo
    for subj in sorted(root_p.iterdir(), key=lambda p: p.name):
        if not subj.is_dir():
            continue
        if not subj.name.lower().startswith("sub"):
            continue

        # caso 1: subXX/anat/skullstripped
        anat_dir = subj / "anat"
        if anat_dir.is_dir():
            skull = anat_dir / "skullstripped"
            if skull.is_dir():
                yield (subj.name, "anat/skullstripped")

        # caso 2: subXX/ses-XX/anat/skullstripped
        for ses in sorted(subj.glob("ses-*"), key=lambda p: p.name):
            ses_anat = ses / "anat"
            if ses_anat.is_dir():
                skull = ses_anat / "skullstripped"
                if skull.is_dir():
                    rel = skull.relative_to(subj)
                    yield (subj.name, str(rel))


# ========================================
# PATH PER LE MODALITÀ (dentro skullstripped)
# ========================================
def path_modality(subject: str, rel_patient: str, modality: str):
    """
    Cerca il file NIfTI della modalità dentro skullstripped.
    """
    folder = os.path.join(base_root, subject, rel_patient)
    if not os.path.isdir(folder):
        return None

    patterns = {
        "FLAIR": ["flair"],
        "T1": ["t1w", "mprage"],
        "T2": ["t2w", "_t2", "-t2", "t2."],
    }

    wanted = patterns.get(modality, [])

    for f in os.listdir(folder):
        f_low = f.lower()

        if not (f_low.endswith(".nii") or f_low.endswith(".nii.gz")):
            continue
        if "mask" in f_low:
            continue

        # se il file contiene sia flair che t2: lo trattiamo come flair
        if "flair" in f_low and "t2" in f_low:
            if modality == "FLAIR":
                return os.path.join(folder, f)
            if modality == "T2":
                continue

        if modality == "T2":
            if "flair" in f_low:
                continue
            if "t2star" in f_low or "t2*" in f_low:
                continue

        if modality == "FLAIR" and "flair" in f_low:
            return os.path.join(folder, f)

        if any(p in f_low for p in wanted):
            return os.path.join(folder, f)

    return None


# ========================================
# OUTPUT LOCALE
# ========================================
def output_root(subject: str, rel_patient: str) -> str:
    return os.path.join(base_root, subject, rel_patient, "Output", "ALL_AXES")


# ========================================
# FUNZIONI DI ELABORAZIONE
# ========================================
def load_and_resample(path, is_mask=False):
    img = nib.load(path)
    order = 0 if is_mask else 1
    resampled_img = resample_to_output(img, voxel_sizes=voxel_sizes, order=order)
    data = resampled_img.get_fdata()
    data[data < 0] = 0
    if not is_mask:
        dmin, dmax = np.nanmin(data), np.nanmax(data)
        data = (data - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(data)
    return data


def compute_crop_and_valid_slices(ref_vol, orientation, nz_threshold, resize_shape):
    if orientation == 'axial':
        mask2d = np.any(ref_vol > 0, axis=2)
        slicer = lambda idx: ref_vol[:, :, idx]
        n_slices = ref_vol.shape[2]
    elif orientation == 'coronal':
        mask2d = np.any(ref_vol > 0, axis=1)
        slicer = lambda idx: ref_vol[:, idx, :]
        n_slices = ref_vol.shape[1]
    elif orientation == 'sagittal':
        mask2d = np.any(ref_vol > 0, axis=0)
        slicer = lambda idx: ref_vol[idx, :, :]
        n_slices = ref_vol.shape[0]
    else:
        raise ValueError("orientation deve essere 'axial', 'coronal' o 'sagittal'.")

    coords = np.argwhere(mask2d)
    if coords.size == 0:
        raise ValueError(f"Nessun voxel non-zero nel volume di riferimento per {orientation}")

    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)

    valid_indices = []
    for i in range(n_slices):
        slice_2d = slicer(i)
        cropped = slice_2d[ymin:ymax+1, xmin:xmax+1]
        resized = resize(cropped, resize_shape, preserve_range=True,
                         anti_aliasing=True, order=1)
        out_uint8 = (resized * 255).astype(np.uint8)
        if np.count_nonzero(out_uint8) >= nz_threshold:
            valid_indices.append(i)

    return (ymin, xmin, ymax, xmax), valid_indices


def save_slices_for_orientation(subject,
                                rel_patient,
                                modality,
                                src_path,
                                vol,
                                orientation,
                                crop_box,
                                valid_indices,
                                out_dir,
                                resize_shape,
                                global_dir=None):
    ymin, xmin, ymax, xmax = crop_box

    base_name = os.path.basename(src_path)
    if base_name.lower().endswith(".nii.gz"):
        base_name = base_name[:-7]
    elif base_name.lower().endswith(".nii"):
        base_name = base_name[:-4]

    if orientation == 'axial':
        slicer = lambda idx: vol[:, :, idx]
    elif orientation == 'coronal':
        slicer = lambda idx: vol[:, idx, :]
    else:
        slicer = lambda idx: vol[idx, :, :]

    os.makedirs(out_dir, exist_ok=True)
    if orientation == "axial" and global_dir:
        os.makedirs(global_dir, exist_ok=True)

    count = 0
    for i in valid_indices:
        sl = slicer(i)
        cropped = sl[ymin:ymax+1, xmin:xmax+1]
        resized = resize(cropped, resize_shape, preserve_range=True, anti_aliasing=True)
        out_slice = (resized * 255).astype(np.uint8)

        fname = f"{base_name}_slice_{i:03d}.png"
        local_path = os.path.join(out_dir, fname)
        imageio.imwrite(local_path, out_slice)

        if orientation == "axial" and global_dir:
            imageio.imwrite(os.path.join(global_dir, fname), out_slice)

        count += 1

    return count


# ========================================
# PROCESS DI UN SOGGETTO/ANAT
# ========================================
def process_subject(subject: str, rel_patient: str, global_crop_valid=None):
    """
    Se global_crop_valid è None, calcola il crop (e lo restituisce).
    Se global_crop_valid NON è None, usa quello per estrarre le stesse slice.
    """
    print(f"\n=== {subject} | {rel_patient} ===")

    skull_dir = os.path.join(base_root, subject, rel_patient)
    if not os.path.isdir(skull_dir):
        print(f"[SKIP] Nessuna cartella skullstripped qui: {skull_dir}")
        return False, global_crop_valid

    # 🧹 pulizia output
    out_root = output_root(subject, rel_patient)
    out_parent = os.path.dirname(out_root)
    if os.path.isdir(out_parent):
        print(f"   🧹 Output esistente trovato ({out_parent}), lo elimino e ricreo...")
        shutil.rmtree(out_parent)
    os.makedirs(out_root, exist_ok=True)

    # cerchiamo i volumi disponibili in skullstripped
    modalities_paths = {}
    for m in ["FLAIR", "T1", "T2"]:
        p = path_modality(subject, rel_patient, m)
        if p is not None and os.path.isfile(p):
            modalities_paths[m] = p

    if not modalities_paths:
        print(f"[SKIP] Nessuna modalità trovata in {skull_dir}")
        return False, global_crop_valid

    # volume di riferimento
    if ref_modality in modalities_paths:
        ref_path = modalities_paths[ref_modality]
    else:
        ref_path = modalities_paths.get("T1") or modalities_paths.get("T2")
        if ref_path is None:
            print(f"[SKIP] Nessuna modalità di riferimento trovata.")
            return False, global_crop_valid

    ref_vol = load_and_resample(ref_path, is_mask=False)

    orientations = ["axial", "coronal", "sagittal"]

    # 👉 se non abbiamo ancora il crop globale, lo creiamo adesso da questo soggetto
    if global_crop_valid is None:
        crop_valid = {}
        for ori in orientations:
            crop_box, valid_idx = compute_crop_and_valid_slices(ref_vol, ori, nz_threshold, resize_shape)
            crop_valid[ori] = (crop_box, valid_idx)
            print(f"[{ori}] Kept {len(valid_idx)} slices (DEFINITO come riferimento globale)")
        global_crop_valid = crop_valid
    else:
        # opzionale: potresti verificare che la shape del volume sia compatibile
        print("Uso crop INDICI GLOBALI già calcolati.")

    # per ogni modalità
    for mod, vol_path in modalities_paths.items():
        mod_dir = os.path.join(out_root, mod)
        os.makedirs(mod_dir, exist_ok=True)

        vol = load_and_resample(vol_path, is_mask=False)

        total_saved = 0
        for ori in orientations:
            crop_box, valid_idx = global_crop_valid[ori]
            ori_dir = os.path.join(mod_dir, ori)
            os.makedirs(ori_dir, exist_ok=True)

            if mod == "T1":
                g_dir = GLOBAL_TRAIN_A
            elif mod == "FLAIR":
                g_dir = GLOBAL_TRAIN_B
            else:
                g_dir = None

            saved = save_slices_for_orientation(
                subject,
                rel_patient,
                mod,
                vol_path,
                vol,
                ori,
                crop_box,
                valid_idx,
                ori_dir,
                resize_shape,
                global_dir=g_dir,
            )
            total_saved += saved
            print(f"Saved {saved} {mod} {ori} slices -> {ori_dir}")

        print(f"=> {mod}: {total_saved} slice totali.")

    return True, global_crop_valid


# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    missing_skull = []
    global_crop_valid = None   # 👈 qui terremo il crop del primo soggetto

    for subject, rel_patient in discover_subject_anat_pairs(base_root):
        ok, global_crop_valid = process_subject(subject, rel_patient, global_crop_valid)
        if not ok:
            missing_skull.append(f"{subject}/{rel_patient}")

    print("\n✅ Estrazione completata.")
    if missing_skull:
        print("\n⚠️ Soggetti/ANAT senza skullstripped (o senza volumi utili):")
        for s in missing_skull:
            print(" -", s)
    else:
        print("Tutti avevano skullstripped e almeno una modalità utile.")

    # ========================================
    # 🔍 REPORT FINALE: confronto trainA / trainB
    # ========================================
    def count_png(folder):
        return sum(1 for f in Path(folder).glob("*.png"))

    count_A = count_png(GLOBAL_TRAIN_A)
    count_B = count_png(GLOBAL_TRAIN_B)

    print("\n📊 Riepilogo finale:")
    print(f"  - trainA (T1)   : {count_A:,} immagini")
    print(f"  - trainB (FLAIR): {count_B:,} immagini")

    if count_A == count_B:
        print("✅ Le due cartelle hanno lo stesso numero di immagini (ottimo).")
    else:
        diff = abs(count_A - count_B)
        print(f"⚠️ Differenza di {diff} immagini tra trainA e trainB.")
