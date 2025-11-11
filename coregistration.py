import os
from pathlib import Path
import SimpleITK as sitk

# ===================== CONFIG =====================

BASE_DATASET = Path(r"E:\Datasets\VOLUMI-SANI-1mm")  # root dataset
MODE = "affine"
SAVE_TFM = True

HIST_BINS = 50
ITERS = 300
SAMPLING_PCT = 0.25
SHRINK = (4, 2, 1)
SMOOTH = (2, 1, 0)

# ===================== FUNZIONI BASE =====================

def read_img(p: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(p), pixel_type)

def write_img(img: sitk.Image, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(p))
    print(f"[SAVED] {p.resolve()}")

def write_tx(tx: sitk.Transform, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(tx, str(p))
    print(f"[SAVED TX] {p.resolve()}")

def build_init_tx(fixed: sitk.Image, moving: sitk.Image, mode: str):
    base = sitk.Euler3DTransform() if mode == "rigid" else sitk.AffineTransform(3)
    return sitk.CenteredTransformInitializer(
        fixed, moving, base, sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

def register(fixed_img: sitk.Image, moving_img: sitk.Image, mode="rigid"):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=HIST_BINS)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(SAMPLING_PCT)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=ITERS,
        relaxationFactor=0.5
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(list(SHRINK))
    R.SetSmoothingSigmasPerLevel(list(SMOOTH))
    try:
        R.SmoothingSigmasAreSpecifiedInPhysicalUnits(False)
    except AttributeError:
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    init_tx = build_init_tx(fixed_img, moving_img, mode)
    R.SetInitialTransform(init_tx, inPlace=False)
    final_tx = R.Execute(fixed_img, moving_img)
    return final_tx, R.GetMetricValue()

def resample_with_tx(moving: sitk.Image, fixed_like: sitk.Image, tx: sitk.Transform,
                     interp=sitk.sitkLinear, default_val=0.0):
    return sitk.Resample(moving, fixed_like, tx, interp, default_val, moving.GetPixelID())

# ===================== FIND MODALITIES =====================

def find_modalities_in_folder(folder: Path):
    t1 = t2 = flair = None
    for p in folder.glob("*.nii*"):
        name = p.name.lower()
        if "t1w" in name or "mprage" in name:
            t1 = p
        elif "t2w" in name and "t2star" not in name and "t2*" not in name:
            t2 = p
        elif "flair" in name:
            flair = p
    return t1, t2, flair

# ===================== CHECK GIÀ FATTO =====================

def already_processed(work_dir: Path, t1_name: str | None) -> bool:
    """
    work_dir: la cartella dove stiamo lavorando (anat o anat/3dVOL)
    t1_name: nome del file T1 che useremmo come riferimento
    Ritorna True se esiste già la cartella di output e (se sappiamo il nome T1)
    c'è già la T1 dentro.
    """
    out_dir = work_dir / "volumi_coregistrati_alla_t1"
    if not out_dir.is_dir():
        return False

    if t1_name is None:
        # non so che T1 cercare, ma la cartella esiste → consideriamo fatto
        return True

    t1_out = out_dir / t1_name
    return t1_out.is_file()

# ===================== PROCESS ANAT =====================

def process_anat_folder(anat_dir: Path):
    if not anat_dir.is_dir():
        print(f"   ❌ {anat_dir} non è una cartella, salto.")
        return

    # se c'è 3dVOL lavoriamo lì
    vol_dir = anat_dir / "3dVOL"
    if vol_dir.is_dir():
        work_dir = vol_dir
        print(f"   📁 Uso la cartella 3dVOL: {work_dir}")
    else:
        work_dir = anat_dir
        print(f"   📁 Uso la cartella anat direttamente: {work_dir}")

    # cerco i volumi
    t1_p, t2_p, flair_p = find_modalities_in_folder(work_dir)

    if t1_p is None:
        print(f"   ❌ Nessuna T1 (t1w / mprage) trovata in {work_dir}, salto.")
        return

    # prima di fare tutto, controlliamo se è già stato fatto
    if already_processed(work_dir, t1_p.name):
        print(f"   ✅ Coregistrazione già presente per {work_dir}, salto.")
        return

    print(f"   ✅ Trovata T1: {t1_p.name}")
    if t2_p:
        print(f"   ✅ Trovata T2: {t2_p.name}")
    else:
        print(f"   ℹ️  Nessuna T2 trovata.")
    if flair_p:
        print(f"   ✅ Trovata FLAIR: {flair_p.name}")
    else:
        print(f"   ℹ️  Nessuna FLAIR trovata.")

    t1_img = read_img(t1_p)

    out_dir = work_dir / "volumi_coregistrati_alla_t1"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"   📂 Cartella di output: {out_dir}")

    # salvo la T1 di riferimento
    write_img(t1_img, out_dir / t1_p.name)

    # T2 → T1
    if t2_p is not None:
        t2_img = read_img(t2_p)
        print("   🔁 Registro T2 → T1 ...")
        t2_tx, t2_metric = register(t1_img, t2_img, mode=MODE)
        print(f"   📏 Metrica T2→T1: {t2_metric:.6f}")
        t2_coreg = resample_with_tx(t2_img, t1_img, t2_tx, interp=sitk.sitkLinear, default_val=0.0)
        write_img(t2_coreg, out_dir / t2_p.name)
        if SAVE_TFM:
            write_tx(t2_tx, out_dir / f"{t2_p.stem}_to_T1.tfm")

    # FLAIR → T1
    if flair_p is not None:
        flair_img = read_img(flair_p)
        print("   🔁 Registro FLAIR → T1 ...")
        flair_tx, flair_metric = register(t1_img, flair_img, mode=MODE)
        print(f"   📏 Metrica FLAIR→T1: {flair_metric:.6f}")
        flair_coreg = resample_with_tx(flair_img, t1_img, flair_tx, interp=sitk.sitkLinear, default_val=0.0)
        write_img(flair_coreg, out_dir / flair_p.name)
        if SAVE_TFM:
            write_tx(flair_tx, out_dir / f"{flair_p.stem}_to_T1.tfm")

# ===================== MAIN =====================

def main():
    base = BASE_DATASET.resolve()
    if not base.exists():
        print(f"❌ Base dataset non trovata: {base}")
        return

    print(f"🔎 Scansiono i soggetti in: {base}")

    # prendi tutte le cartelle che iniziano con "sub" (sub01, sub1, sub-ON...)
    sub_dirs = [p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith("sub")]

    # ordina per numero dopo "sub"
    def sub_key(p: Path):
        name = p.name.lower()
        rest = name[3:]
        rest = rest.lstrip("-")
        try:
            return int(rest)
        except ValueError:
            return 999999

    sub_dirs = sorted(sub_dirs, key=sub_key)

    if not sub_dirs:
        print("⚠️ Nessuna cartella che inizi con 'sub'")
        return

    for sub_dir in sub_dirs:
        print(f"\n📂 Soggetto: {sub_dir.name}")

        ses_dirs = list(sub_dir.glob("ses-*"))
        if not ses_dirs:
            print("   ⚠️ Nessuna cartella 'ses-*' trovata.")
            anat_dir = sub_dir / "anat"
            if anat_dir.is_dir():
                print(f"   ✅ Trovata cartella anat: {anat_dir}")
                process_anat_folder(anat_dir)
            else:
                print(f"   ❌ Nessuna cartella anat trovata in {sub_dir}")
            continue

        for ses_dir in ses_dirs:
            print(f"   📁 Sessione: {ses_dir.name}")
            anat_dir = ses_dir / "anat"
            if anat_dir.is_dir():
                print(f"      ✅ Trovata anat: {anat_dir}")
                process_anat_folder(anat_dir)
            else:
                print(f"      ❌ Nessuna cartella anat trovata in {ses_dir}")

    print("\n✅ Coregistrazione completata per tutte le cartelle trovate.")

if __name__ == "__main__":
    main()
