"""Error class for handled errors."""


class DataError(Exception):
    """
    Base class for data-related exceptions.
    
    This exception is raised for errors that occur during data handling.
    """
    pass


class FileExistsError(DataError):
    """
    Exception raised when a file already exists.
    
    This error is raised to indicate that an operation cannot proceed
    because a file with the specified name already exists.
    """
    pass


class MmcifParsingError(DataError):
    """
    Exception raised when parsing of MMCIF (Macromolecular Crystallographic Information File) fails.
    
    This error is raised when there is a failure in parsing an MMCIF file,
    which is commonly used to store structural data of molecules.
    """
    pass


class ResolutionError(DataError):
    """
    Exception raised when the resolution is not acceptable.
    
    This error is raised when the resolution value does not meet the required criteria.
    The resolution could refer to the detail level in structural data or other context-specific measurements.
    """
    pass


class LengthError(DataError):
    """
    Exception raised when the length is not acceptable.
    
    This error is raised when a length value, such as the length of a sequence or structure,
    does not meet the specified requirements.
    """
    pass