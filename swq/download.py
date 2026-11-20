import hashlib
import os
import sys
import urllib.request
import zipfile

from swq import data as D

ZENODO = "https://zenodo.org/api/records/{}/files/{}/content"
FILES = [
    ("CoastDOM", "https://doi.pangaea.de/10.1594/PANGAEA.964012?format=textfile", D.COAST_FILE, "9ac81bdc69005cd61583f6e5e62b370e"),
    ("USGS SSC", ZENODO.format(22298970, D.SSC_FILE), D.SSC_FILE, "c90246ba18572d9600b79a772bebe32d"),
    ("Caravan-Qual", ZENODO.format(21337532, "wqms_site_info.csv"), "caravan/wqms_site_info.csv", "e572108191c49b838085ebb58fe637e1"),
    ("Caravan-Qual", ZENODO.format(21337532, "wqms-csv.zip"), "caravan/wqms-csv.zip", "b093f92627f671a4f8be366838a6996b"),
    ("Caravan-Qual", ZENODO.format(21337532, "Caravan-Qual_lite.zarr.zip"), "caravan/Caravan-Qual_lite.zarr.zip", "3c3bbee1dc1f5f81df189d80e14e6a77"),
]
GLORIA_URL = "https://download.pangaea.de/dataset/948492/files/GLORIA-2022.zip"
GLORIA_DIR = os.path.join(D.DATA, "GLORIA_2022")
GLORIA_MD5 = {
    "GLORIA_meta_and_lab.csv": "457a7b1c4ae6304e1b449255610e7d4a",
    "GLORIA_Rrs.csv": "0ba53ce6e3eb58d35f85733cd9209d18",
    "GLORIA_qc_flags.csv": "373aba4d24aa13a2ceb35afb6757e726",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "swq-download"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(dest + ".part", dest)


def gloria():
    if not all(os.path.exists(os.path.join(GLORIA_DIR, n)) for n in GLORIA_MD5):
        zp = os.path.join(D.DATA, "GLORIA-2022.zip")
        print("GLORIA: downloading", GLORIA_URL, flush=True)
        fetch(GLORIA_URL, zp)
        os.makedirs(GLORIA_DIR, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for m in z.infolist():
                name = os.path.basename(m.filename)
                if name in GLORIA_MD5:
                    with z.open(m) as src, open(os.path.join(GLORIA_DIR, name), "wb") as dst:
                        dst.write(src.read())
        os.remove(zp)
    ok = True
    for n, h in GLORIA_MD5.items():
        good = md5(os.path.join(GLORIA_DIR, n)) == h
        print("GLORIA", n, "ok" if good else "CHECKSUM MISMATCH")
        ok &= good
    return ok


def other():
    ok = True
    for name, url, rel, h in FILES:
        path = os.path.join(D.DATA, rel)
        if not os.path.exists(path):
            print(f"{name}: downloading {rel}", flush=True)
            fetch(url, path)
        good = md5(path) == h
        print(name, rel, "ok" if good else "CHECKSUM MISMATCH")
        ok &= good
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([gloria(), other()]) else 1)
