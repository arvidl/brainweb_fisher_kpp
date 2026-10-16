"""Restore official BrainWeb data; no parent repository paths or FreeSurfer.

The normal T1 image and discrete tissue phantom are downloaded as MINC then
converted to canonical RAS NIfTI using nibabel. Optional identity form fields
are left empty. Bundled NIfTI files mean no download is needed for normal use.
"""
from pathlib import Path
import gzip
import hashlib
import json
import tempfile
import urllib.parse
import urllib.request

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1"
VOLUMES = {
    "t1_icbm_normal_1mm_pn0_rf0.nii.gz": "T1 ICBM normal 1mm pn0 rf0",
    "phantom_1.0mm_normal_crisp.nii.gz": "phantom_1.0mm_normal_crisp",
}


def main():
    DATA.mkdir(exist_ok=True)
    old = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else None
    files = {}
    for filename, alias in VOLUMES.items():
        destination = DATA / filename
        if not destination.exists():
            print(f"Fetching {alias}", flush=True)
            body = urllib.parse.urlencode({"do_download_alias": alias, "format_value": "minc",
                                           "zip_value": "gnuzip", "download_for_real": "[Start download!]"}).encode()
            request = urllib.request.Request(URL, data=body)
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            if payload[:2] != b"\x1f\x8b":
                raise RuntimeError("BrainWeb did not return gzip data; inspect the official download form.")
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "volume.mnc"
                source.write_bytes(gzip.decompress(payload))
                image = nib.as_closest_canonical(nib.load(source))
                dtype = np.uint8 if "crisp" in filename else np.float32
                values = np.asarray(image.dataobj, dtype=dtype)
                converted = nib.Nifti1Image(values, image.affine)
                converted.header.set_xyzt_units("mm")
                nib.save(converted, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        files[filename] = {"sha256": digest, "bytes": destination.stat().st_size, "brainweb_alias": alias}
        # A fresh MINC -> NIfTI conversion can encode identically valued data with
        # different headers. Compare voxel values rather than overwrite provenance.
        if old and filename in old["files"] and old["files"][filename]["sha256"] != digest:
            print(f"New encoding/checksum for {filename}; inspect before distributing.")
    manifest = {"source": URL, "model": "BrainWeb original normal phantom (not the 20-subject models)",
                "anatomy": "https://brainweb.bic.mni.mcgill.ca/anatomic_normal.html", "files": files}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Data ready.")


if __name__ == "__main__":
    main()
