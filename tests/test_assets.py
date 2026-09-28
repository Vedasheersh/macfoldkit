import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "src/macfoldkit/runners/fetch_assets.py"
spec = importlib.util.spec_from_file_location("macfoldkit_fetch_test", path)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetTests(unittest.TestCase):
    def test_checksum_mismatch_preserves_existing_file_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "asset"
            target.write_bytes(b"old")
            with patch.object(assets.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    assets.download_verified("https://example.org/file", target, hashlib.sha256(b"expected").hexdigest())
            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_verified_cache_does_not_contact_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "asset"
            target.write_bytes(b"good")
            with patch.object(assets.urllib.request, "urlopen") as network:
                assets.download_verified("https://example.org/file", target, hashlib.sha256(b"good").hexdigest())
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
