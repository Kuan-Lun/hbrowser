"""判斷例外是否屬於底層連線中斷（例如 CDP websocket 斷線）。"""

from websockets.exceptions import ConnectionClosed

_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (ConnectionClosed,)


def is_connection_error(exc: BaseException) -> bool:
    """連線層例外必須往外傳給上層的重連/重啟邏輯處理，不能被重試迴圈裡的
    `except Exception` 當成「元素還沒出現」之類的暫時性錯誤吞掉。
    """
    return isinstance(exc, _CONNECTION_ERRORS)
