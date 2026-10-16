class GoalCancelledException(BaseException):
    """Raised when an active LLM request cancels its owning goal.

    This is a control-flow sentinel so generic error handlers do not convert
    cancellation into an ordinary goal failure before the parallel boundary.
    """

    def __init__(self, message: str, cancellation_token: str | None = None):
        super().__init__(message)
        self.cancellation_token = cancellation_token
