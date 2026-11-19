"""Checks for failures before a downloaded CPU package can run."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from verify_artifact import verify_artifact


class VerifyArtifactTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.download = root / 'download'
        self.dist = root / 'dist'
        self.download.mkdir()
        self.dist.mkdir()
        self.archive = 'nlp-test-target.zip'
        self.checksum = self.archive + '.sha256'
        (self.dist / self.checksum).write_bytes(b'local checksum')
        (self.dist / self.archive).write_bytes(b'local package')

    def upload(self, members):
        artifact = self.download / 'artifact'  # The service may omit the file extension.
        with zipfile.ZipFile(artifact, 'w') as outer:
            for name, data in members:
                outer.writestr(name, data)
        return hashlib.sha256(artifact.read_bytes()).hexdigest()

    def verify(self, digest):
        verify_artifact(self.download, digest, self.dist, 'test-target', 'test-commit', 'native-bins')

    def test_receipt_and_downloaded_bytes_reach_native_verifier(self):
        digest = self.upload([(self.archive, b'downloaded package'), (self.checksum, b'downloaded checksum')])

        def native_verify(command, check):
            self.assertTrue(check)
            self.assertEqual(Path(command[1]).name, 'verify.py')
            self.assertEqual(Path(command[2]).read_bytes(), b'downloaded package')
            self.assertEqual(Path(command[3]).read_bytes(), b'downloaded checksum')
            self.assertEqual(command[4:], ['test-target', 'test-commit', 'native-bins'])

        with patch('verify_artifact.subprocess.run', side_effect=native_verify) as verifier:
            self.verify(digest)
            verifier.assert_called_once()

    def test_missing_or_extra_download_fails(self):
        with self.assertRaisesRegex(ValueError, 'single downloaded'):
            self.verify('0' * 64)
        digest = self.upload([])
        (self.download / 'extra').touch()
        with self.assertRaisesRegex(ValueError, 'single downloaded'):
            self.verify(digest)

    def test_mismatched_upload_digest_fails(self):
        self.upload([(self.archive, b'package'), (self.checksum, b'checksum')])
        with self.assertRaisesRegex(ValueError, 'SHA-256 differs'):
            self.verify('0' * 64)

    def test_missing_and_unexpected_members_fail(self):
        for names in [[self.checksum], [self.archive, self.checksum, 'extra']]:
            with self.subTest(names=names):
                digest = self.upload([(name, b'x') for name in names])
                with self.assertRaisesRegex(ValueError, 'exactly the native package'):
                    self.verify(digest)

    def test_native_checksum_failure_propagates(self):
        receipt = ('0' * 64 + '  ' + self.archive + '\n').encode()
        digest = self.upload([(self.archive, b'package'), (self.checksum, receipt)])
        command = [sys.executable, str(Path(__file__).with_name('verify_artifact.py')),
                   str(self.download), digest, str(self.dist), 'test-target', 'test-commit', 'native-bins']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('verify.py', result.stderr)
        self.assertIn('AssertionError', result.stderr)
        self.assertNotIn('smoke.py', result.stderr)
