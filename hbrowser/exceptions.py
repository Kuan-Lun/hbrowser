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


class CaptchaAPIKeyNotSetException(Exception):
    def __init__(
        self, message: str = "APIKEY_2CAPTCHA environment variable is not set."
    ) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class CaptchaSolveException(Exception):
    def __init__(self, message: str = "Failed to solve CAPTCHA challenge.") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
