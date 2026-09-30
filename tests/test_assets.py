import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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

    def test_huggingface_asset_uses_configured_endpoint_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = Path(directory) / "cached"
            cached.write_bytes(b"verified")
            target = Path(directory) / "asset"
            hub_download = Mock(return_value=str(cached))
            endpoint = "https://mirror.example.org/hub"
            with patch.dict(assets.os.environ, {"HF_ENDPOINT": endpoint}), \
                 patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=hub_download)}), \
                 patch.object(assets.urllib.request, "urlopen") as network:
                assets.download_verified(assets.MOLS_URL, target, hashlib.sha256(b"verified").hexdigest())
            self.assertEqual(target.read_bytes(), b"verified")
            hub_download.assert_called_once_with(repo_id="boltz-community/boltz-2",
                                                 revision="6fdef46d763fee7fbb83ca5501ccceff43b85607",
                                                 filename="mols.tar", endpoint=endpoint)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
