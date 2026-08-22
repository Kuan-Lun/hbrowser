import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from hbrowser.gallery.browser import chrome_manager
from hbrowser.gallery.utils import Deadline


def _version_info() -> dict[str, object]:
    return {
        "version": "123.0.0",
        "downloads": {
            "chrome": [
                {
                    "platform": "linux64",
                    "url": "https://example.invalid/chrome.zip",
                }
            ]
        },
    }


def _create_complete_install(root: Path, *, contents: str = "chrome") -> Path:
    version_dir = root / "123.0.0"
    chrome = version_dir / "chrome-linux64" / "chrome"
    chrome.parent.mkdir(parents=True)
    chrome.write_text(contents, encoding="utf-8")
    (version_dir / chrome_manager._INSTALL_MARKER_FILENAME).write_text(
        json.dumps(
            {
                "schema": chrome_manager._INSTALL_MARKER_SCHEMA,
                "version": "123.0.0",
            }
        ),
        encoding="utf-8",
    )
    return chrome


class ChromeCachePublicationTests(unittest.TestCase):
    def _patch_platform(self, cache_dir: Path) -> tuple[Any, ...]:
        return (
            patch.object(chrome_manager, "_get_cache_dir", return_value=cache_dir),
            patch.object(chrome_manager, "get_platform", return_value="linux64"),
            patch.object(
                chrome_manager,
                "get_chrome_executable_name",
                return_value="chrome",
            ),
            patch.object(
                chrome_manager,
                "_fetch_stable_version_info",
                return_value=_version_info(),
            ),
            patch(
                "hbrowser.gallery.browser.chrome_manager.platform.system",
                return_value="Linux",
            ),
        )

    def test_complete_marker_is_required_for_cached_install(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-chrome-cache-") as directory:
            cache_dir = Path(directory)
            chrome = _create_complete_install(cache_dir)
            patches = self._patch_platform(cache_dir)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patch.object(chrome_manager, "_download_and_extract") as download,
            ):
                result = chrome_manager.ensure_chrome_installed(
                    deadline=Deadline.after(1)
                )

            self.assertEqual(result.chrome, str(chrome))
            download.assert_not_called()

    def test_staging_is_published_only_after_validation_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-chrome-stage-") as directory:
            cache_dir = Path(directory)
            patches = self._patch_platform(cache_dir)

            def download(
                _: str,
                destination: Path,
                __: str,
                *,
                deadline: Deadline,
            ) -> None:
                self.assertFalse(deadline.expired)
                chrome = destination / "chrome-linux64" / "chrome"
                chrome.parent.mkdir(parents=True)
                chrome.write_text("new", encoding="utf-8")

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patch.object(
                    chrome_manager,
                    "_download_and_extract",
                    side_effect=download,
                ),
                patch.object(chrome_manager, "_make_all_files_executable"),
                patch.object(chrome_manager, "_remove_quarantine"),
            ):
                staging_root = chrome_manager.create_chrome_install_staging_root()
                result = chrome_manager.ensure_chrome_installed(
                    deadline=Deadline.after(1),
                    staging_root=staging_root,
                )

            chrome = Path(result.chrome)
            self.assertEqual(chrome.read_text(encoding="utf-8"), "new")
            self.assertTrue(
                chrome_manager._installation_is_complete(
                    cache_dir / "123.0.0",
                    chrome,
                    version="123.0.0",
                )
            )
            self.assertFalse(staging_root.exists())

    def test_legacy_orphan_is_removed_under_the_next_install_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-chrome-orphan-") as directory:
            cache_dir = Path(directory)
            chrome = _create_complete_install(cache_dir)
            orphan = cache_dir / ".123.0.0.staging-interrupted"
            orphan.mkdir()
            (orphan / "partial-download").write_text("partial", encoding="utf-8")
            patches = self._patch_platform(cache_dir)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                result = chrome_manager.ensure_chrome_installed(
                    deadline=Deadline.after(1)
                )

            self.assertEqual(Path(result.chrome), chrome)
            self.assertFalse(orphan.exists())

    def test_failed_staging_does_not_replace_previous_complete_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-chrome-rollback-"
        ) as directory:
            cache_dir = Path(directory)
            chrome = _create_complete_install(cache_dir, contents="old")
            patches = self._patch_platform(cache_dir)

            def download(
                _: str,
                destination: Path,
                __: str,
                *,
                deadline: Deadline,
            ) -> None:
                self.assertFalse(deadline.expired)
                staged = destination / "chrome-linux64" / "chrome"
                staged.parent.mkdir(parents=True)
                staged.write_text("new", encoding="utf-8")

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patch.object(
                    chrome_manager,
                    "_download_and_extract",
                    side_effect=download,
                ),
                patch.object(chrome_manager, "_make_all_files_executable"),
                patch.object(
                    chrome_manager,
                    "_remove_quarantine",
                    side_effect=TimeoutError("worker cutoff"),
                ),
                self.assertRaisesRegex(TimeoutError, "worker cutoff"),
            ):
                chrome_manager.ensure_chrome_installed(
                    force_download=True,
                    deadline=Deadline.after(1),
                )

            self.assertEqual(chrome.read_text(encoding="utf-8"), "old")
            self.assertTrue(
                chrome_manager._installation_is_complete(
                    cache_dir / "123.0.0",
                    chrome,
                    version="123.0.0",
                )
            )

    def test_valid_backup_recovers_interrupted_publication_without_download(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-chrome-recover-"
        ) as directory:
            cache_dir = Path(directory)
            chrome = _create_complete_install(cache_dir)
            backup = cache_dir / ".123.0.0.previous"
            (cache_dir / "123.0.0").rename(backup)
            patches = self._patch_platform(cache_dir)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patch.object(chrome_manager, "_download_and_extract") as download,
            ):
                result = chrome_manager.ensure_chrome_installed(
                    deadline=Deadline.after(1)
                )

            self.assertEqual(Path(result.chrome), chrome)
            self.assertTrue(chrome.is_file())
            self.assertFalse(backup.exists())
            download.assert_not_called()
