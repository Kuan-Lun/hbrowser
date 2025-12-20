"""URL 相關工具函數"""


def matchurl(*args: str) -> bool:
    """
    比較多個 URL 是否相同（忽略結尾的斜線）

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
    fixargs = list()
    for url in args:
        # 處理 None 或空字串的情況
        if not url:
            return False
        while url[-1] == "/":
            url = url[0:-1]
        fixargs.append(url)

    t = True
    for url in fixargs[1:]:
        t &= fixargs[0] == url
    return t
