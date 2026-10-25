"""URL 相關工具函數"""


def matchurl(*args: str) -> bool:
    """
    比較多個 URL 是否相同（忽略結尾的斜線）

    根路徑與全斜線字串視為同一路徑；空字串不匹配任何 URL。

    Example:
        matchurl("https://e-hentai.org", "https://e-hentai.org/") -> True
        matchurl("https://e-hentai.org", "https://e-hentai.org") -> True
        matchurl("https://e-hentai.org", "https://exhentai.org") -> False
        matchurl(
            "https://e-hentai.org",
            "https://e-hentai.org",
            "https://e-hentai.org"
        ) -> True
    """
    fixargs = []
    for url in args:
        # 處理 None 或空字串的情況
        if not url:
            return False
        # A root-relative URL remains a root after removing trailing slashes.
        fixargs.append(url.rstrip("/") or "/")

    t = True
    for url in fixargs[1:]:
        t &= fixargs[0] == url
    return t
