"""數據模型類"""


class Tag:
    """Gallery 標籤"""

    def __init__(
        self,
        filter: str,
        name: str,
        href: str,
    ) -> None:
        self.filter = filter
        self.name = name
        self.href = href

    def __repr__(self) -> str:
        itemlist = [
            f"{attr_name}: {attr_value}"
            for attr_name, attr_value in self.__dict__.items()
        ]
        return "\n".join(itemlist)

    def __str__(self) -> str:
        return ", ".join(self.__repr__().split("\n"))
