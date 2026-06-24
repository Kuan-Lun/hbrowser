class ClientOfflineException(Exception):
    def __init__(self, message: str = "H@H client appears to be offline.") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InsufficientFundsException(Exception):
    def __init__(
        self, message: str = "Insufficient funds to start the download."
    ) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class LoginFailedException(Exception):
    def __init__(self, message: str = "Login did not succeed.") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
