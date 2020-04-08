class ApplicationError(Exception):
    """
    General Error during prediction
    """
    pass


class CouldNotReadImageError(ApplicationError):
    """
    Error reading input image.
    """
    pass


class FailedToWriteResultsError(ApplicationError):
    """
    Error writing results
    """
    pass
