class ClientOfflineException(Exception):
    def __init__(self, message="H@H client appears to be offline."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class InsufficientFundsException(Exception):
    def __init__(self, message="Insufficient funds to start the download."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class CaptchaAPIKeyNotSetException(Exception):
    def __init__(self, message="APIKEY_2CAPTCHA environment variable is not set."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class CaptchaSolveException(Exception):
    def __init__(self, message="Failed to solve CAPTCHA challenge."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message
