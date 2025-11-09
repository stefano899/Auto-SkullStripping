from pathlib import Path
import re

# cartella di partenza
# se hai path lunghi su Windows puoi mettere: Path(r"\\?\E:\Datasets\Volumi_sani_1mm_MNI")
ROOT = Path(r"E:\Datasets\VOLUMI-SANI-1mm")

# metti a False quando vuoi davvero rinominare
DRY_RUN = False


def info_from_name(name: str):
    """
    Ritorna (has_flair, has_t2) guardando solo lo stem.
    Gestisce T2 e T2w.
    """
    has_flair = re.search(r"flair", name, flags=re.IGNORECASE) is not None
    has_t2 = re.search(r"t2w?", name, flags=re.IGNORECASE) is not None
    return has_flair, has_t2


def new_name_for_both(stem: str) -> str:
    """
    Se nel nome ci sono sia T2/T2w che FLAIR, togli T2/T2w.
    Esempio: ..._T2w_FLAIR_... -> ..._FLAIR_...
    """
    return re.sub(r"[_-]?t2w?", "", stem, flags=re.IGNORECASE)


def new_name_t2_to_flair(stem: str) -> str:
    """
    Sostituisci T2/T2w con FLAIR.
    Esempio: ..._T2w_... -> ..._FLAIR_...
    """
    return re.sub(r"t2w?", "FLAIR", stem, flags=re.IGNORECASE)


def handle_skull_dir(skull_dir: Path, dry_run: bool):
    """Gestisce una singola cartella skullstripped."""
    if not skull_dir.exists():
        return

    try:
        # prendo tutti i file (ordinati per nome per avere uscite pulite)
        files = sorted([f for f in skull_dir.iterdir() if f.is_file()], key=lambda p: p.name)
    except Exception as e:
        print(f"  ERRORE nel leggere i file di {skull_dir}: {e}")
        return

    # 1) capisco se in questa cartella c'è già un FLAIR puro
    has_pure_flair = False
    pure_flair_name = None
    for f in files:
        suffixes = "".join(f.suffixes)
        stem = f.name[:-len(suffixes)] if suffixes else f.name
        has_flair, has_t2 = info_from_name(stem)
        if has_flair and not has_t2:
            has_pure_flair = True
            pure_flair_name = f.name
            break

    print(f"\n--- Controllo: {skull_dir} ---")

    # 2) ora processiamo tutti i file
    for f in files:
        try:
            suffixes = "".join(f.suffixes)
            stem = f.name[:-len(suffixes)] if suffixes else f.name
            has_flair, has_t2 = info_from_name(stem)

            # caso 1: nel nome c'è sia T2 che FLAIR -> togli T2
            if has_flair and has_t2:
                new_stem = new_name_for_both(stem)
                new_path = f.with_name(new_stem + suffixes)

                if new_path.exists():
                    print(f"ATTENZIONE: non rinomino {f.name} → {new_path.name} perché esiste già.")
                    continue

                print(f"Rinomino (FLAIR+T2): {f.name} → {new_path.name}")
                if not dry_run:
                    f.rename(new_path)
                continue

            # caso 2: solo T2/T2w
            if has_t2 and not has_flair:
                if has_pure_flair:
                    # c'è già un FLAIR nella cartella → non tocco questo T2
                    flair_info = f" (FLAIR trovato: {pure_flair_name})" if pure_flair_name else ""
                    print(f"Skip (c'è già un FLAIR){flair_info}: {f.name}")
                    continue

                # non c'è FLAIR nella cartella → rinomino T2 in FLAIR
                new_stem = new_name_t2_to_flair(stem)
                new_path = f.with_name(new_stem + suffixes)

                if new_path.exists():
                    print(f"ATTENZIONE: non rinomino {f.name} → {new_path.name} perché esiste già.")
                    continue

                print(f"Rinomino (T2→FLAIR): {f.name} → {new_path.name}")
                if not dry_run:
                    f.rename(new_path)
                continue

            # caso 3: già solo FLAIR o altro → non fare nulla
            # print(f"OK: {f.name}")
            continue

        except Exception as e:
            print(f"  ERRORE su file {f}: {e}")
            continue


def main():
    if not ROOT.exists():
        print(f"La cartella {ROOT} non esiste.")
        return

    # ordina i soggetti
    subjects = sorted([p for p in ROOT.glob("sub*") if p.is_dir()], key=lambda p: p.name)
    print(f"Trovati {len(subjects)} soggetti.")

    processed = 0

    for sub in subjects:
        try:
            print(f"\n=== Soggetto: {sub.name} ===")

            # prendo eventuali ses-* ordinati
            ses_dirs = sorted([p for p in sub.glob("ses-*") if p.is_dir()], key=lambda p: p.name)

            # caso con sessioni
            for ses in ses_dirs:
                skull_dir = ses / "anat"
                handle_skull_dir(skull_dir, DRY_RUN)

            # caso senza sessioni: subXXX/anat/skullstripped
            if not ses_dirs:
                skull_dir = sub / "anat"
                handle_skull_dir(skull_dir, DRY_RUN)

            processed += 1

        except Exception as e:
            # non blocchiamo tutto se un soggetto ha un problema
            print(f"ERRORE sul soggetto {sub}: {e}")
            continue

    print(f"\nProcessati {processed} soggetti.")
    print("\nFatto." if not DRY_RUN else "\nDRY RUN concluso (nessun file rinominato).")


if __name__ == "__main__":
    main()
