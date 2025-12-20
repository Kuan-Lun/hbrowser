"""ExHentai Driver 實現"""

from .eh_driver import EHDriver


class ExHDriver(EHDriver):
    """ExHentai Driver"""

    def _setname(self) -> str:
        return "ExHentai"
