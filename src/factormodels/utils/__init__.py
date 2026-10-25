"""Local resource and output configuration utilities."""
from .local_path import (get_paths, get_resource_path, get_output_path,
                         resolve_resource_path, resolve_output_path, create_output_directory)

__all__ = [
    'get_paths',
    'get_resource_path',
    'get_output_path',
    'resolve_resource_path',
    'resolve_output_path',
    'create_output_directory',
]
