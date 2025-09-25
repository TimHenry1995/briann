from pathlib import PureWindowsPath, PurePosixPath
import os

def map_path_to_os(path: str) -> str:
    """Maps the given folder path to the appropriate format for the current operating system.
    :param folder_path: The folder path to map.
    :return: The mapped folder path.
    :rtype: str
    """
    
    # Check inptu validity
    if path is None:
        raise ValueError("The folder_path cannot be None.")
    if not isinstance(path, str):
        raise TypeError(f"The folder_path was expected to be a string but is {type(path)}.")
    
    # Map path to operating system
    if os.name == 'nt': # Windows
        path = str(PureWindowsPath(path))
    elif os.name == 'posix':
        path = str(PurePosixPath(path))
        
    # Output
    return path


