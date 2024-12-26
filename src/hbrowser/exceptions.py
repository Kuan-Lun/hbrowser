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
