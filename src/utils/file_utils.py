import os
import re
import tiktoken
from typing import Any


def load_ignore_patterns() -> list[str]:
    """
    Load ignore patterns from .gitignore file or return default patterns.
    
    Returns:
        List of ignore patterns
    """
    try:
        with open('.gitignore', 'r') as file:
            return [
                line.strip() for line in file
                if line.strip() and not line.strip().startswith('#')
            ]
    except (FileNotFoundError, IOError):
        # Default ignore patterns if .gitignore not found
        return [
            'node_modules',
            '.git',
            '.next',
            'dist',
            'build',
            '.env',
            '*.log',
            'venv',
            '__pycache__',
            '*.pyc'
        ]


def should_ignore(path: str, ignore_patterns: list[str]) -> bool:
    """
    Check if a path should be ignored based on ignore patterns.
    
    Args:
        path: File or directory path to check
        ignore_patterns: List of ignore patterns
        
    Returns:
        True if path should be ignored, False otherwise
    """
    for pattern in ignore_patterns:
        if pattern.endswith('/'):
            if pattern in path:
                return True
        elif pattern.startswith('*.'):
            ext = pattern[1:]
            if path.endswith(ext):
                return True
        elif pattern in path:
            return True
    return False


def get_token_count(content: str) -> int:
    """
    Calculate the number of tokens in the content using tiktoken.
    
    Args:
        content: Text content to count tokens for
        
    Returns:
        Number of tokens in the content
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding
    return len(encoding.encode(content))


def get_files_info(directory: str, extensions: list[str], ignore_patterns: list[str]) -> list[dict[str, Any]]:
    """
    Recursively scan a directory for files with specified extensions and calculate token count.
    
    Args:
        directory: Directory to scan
        extensions: List of file extensions to include (e.g. ['.md', '.py'])
        ignore_patterns: List of patterns to ignore
        
    Returns:
        List of dictionaries with file information (path and token count)
    """
    files = []
    
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, '.')
            
            if should_ignore(relative_path, ignore_patterns):
                continue
                
            ext = os.path.splitext(filename)[1]
            if ext in extensions:
                try:
                    with open(full_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    token_count = get_token_count(content)
                    files.append({
                        'file_path': relative_path,
                        'token_count': token_count
                    })
                except (UnicodeDecodeError, IOError) as e:
                    print(f"Error reading file {full_path}: {e}")
    
    return files


def calculate_chunks(token_count: int) -> dict[str, Any]:
    """
    Calculate how to chunk a file based on its token count.
    
    Args:
        token_count: Number of tokens in the file
        
    Returns:
        Dictionary with target size and number of chunks
    """
    if token_count <= 650:
        return {"target_size": token_count, "num_chunks": 1}
    if token_count <= 1500:
        return {"target_size": token_count // 2, "num_chunks": 2}
    if token_count <= 2500:
        return {"target_size": token_count // 3, "num_chunks": 3}
    if token_count <= 4000:
        return {"target_size": token_count // 4, "num_chunks": 4}
    
    print(f"⚠️ File is very large ({token_count} tokens). Splitting into smaller chunks.")
    num_chunks = token_count // 700
    return {"target_size": 700, "num_chunks": num_chunks} 