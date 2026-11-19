#!/usr/bin/env python3
"""Bind native package verification to the exact artifact ZIP uploaded by this job."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def verify_artifact(download_dir, upload_digest, dist, target, commit, release_dir):
    files = list(download_dir.iterdir())
    if len(files) != 1 or files[0].is_symlink() or not files[0].is_file():
        raise ValueError('expected a single downloaded artifact file')
    artifact = files[0]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != upload_digest:
        raise ValueError('downloaded artifact SHA-256 differs from the upload')

    # Use the original package names. All bytes passed to verify.py must come from the download.
    checksums = list(dist.glob('*.sha256'))
    if len(checksums) != 1:
        raise ValueError('expected one native package checksum in the upload')
    checksum = checksums[0].name
    archive = checksum[:-len('.sha256')]
    with zipfile.ZipFile(artifact) as outer, tempfile.TemporaryDirectory() as temp:
        if sorted(outer.namelist()) != sorted([archive, checksum]):
            raise ValueError('artifact must contain exactly the native package and its checksum')
        package = Path(temp) / archive
        receipt = Path(temp) / checksum
        package.write_bytes(outer.read(archive))
        receipt.write_bytes(outer.read(checksum))
        subprocess.run([sys.executable, str(Path(__file__).with_name('verify.py')),
                        str(package), str(receipt), target, commit, release_dir], check=True)


if __name__ == '__main__':
    directory, digest, dist, target, commit, release_dir = sys.argv[1:]
    verify_artifact(Path(directory), digest, Path(dist), target, commit, release_dir)
