class ChurnServiceError(Exception):
    def __init__(self, status_code, code, message, details=None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


