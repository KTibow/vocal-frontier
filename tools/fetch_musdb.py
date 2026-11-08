"""Pull the MUSDB18-HQ test split's mixture + vocals out of the 22.7 GB Zenodo zip with HTTP range reads.

Only the members we need are downloaded (~3.4 GB compressed), stored as FLAC. Zenodo gives ~0.4 MB/s per
connection, so tracks are fetched by parallel workers, one connection each.

    uv run tools/fetch_musdb.py data/musdb [split=test]
"""
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
from remotezip import RemoteZip

URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
STEMS = ("mixture", "vocals")


def fetch(out, t, split):
    with RemoteZip(URL) as z:
        d = os.path.join(out, t.replace(" ", "_"))
        os.makedirs(d, exist_ok=True)
        for s in STEMS:
            dst = os.path.join(d, f"{s}.flac")
            if os.path.exists(dst):
                continue
            data, sr = sf.read(io.BytesIO(z.read(f"{split}/{t}/{s}.wav")), dtype="int16")
            sf.write(dst + ".tmp", data, sr, format="FLAC")
            os.replace(dst + ".tmp", dst)
    print(t, flush=True)


def main(out, split="test", workers=2):
    os.makedirs(out, exist_ok=True)
    with RemoteZip(URL) as z:
        tracks = sorted({n.split("/")[1] for n in z.namelist() if n.startswith(split + "/") and n.endswith(".wav")})
    print(f"{len(tracks)} {split} tracks", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda t: fetch(out, t, split), tracks))


if __name__ == "__main__":
    main(*sys.argv[1:])
