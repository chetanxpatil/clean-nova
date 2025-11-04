#!/usr/bin/env python3

"""
A script to analyze Python import dependencies in a project.
It walks a directory, parses all .py files, and builds a dependency graph,
separating internal (project) imports from external (library) imports.
"""

import os
import ast
from collections import defaultdict
from pathlib import Path

# --- Configuration ---
# Directories to exclude from the analysis
EXCLUDE_DIRS = {'venv', '.venv', '.git', '__pycache__', 'build', 'dist', '.vscode'}


# ---------------------

class ImportVisitor(ast.NodeVisitor):
    """
    An AST visitor to find all import statements in a Python file.
    """

    def __init__(self):
        self.imports = set()
        self.relative_imports = set()

    def visit_Import(self, node):
        """Catches 'import pandas' or 'import my_app.utils'"""
        for alias in node.names:
            # We only care about the top-level module
            # e.g., 'pandas.DataFrame' -> 'pandas'
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Catches 'from pandas import DataFrame' or 'from . import utils'"""
        if node.level > 0:
            # This is a relative import (e.g., 'from .utils import ...')
            # node.module is 'utils' and node.level is 1 for '.'
            module_name = "." * node.level
            if node.module:
                module_name += node.module
            self.relative_imports.add(module_name)
        elif node.module:
            # This is an absolute import (e.g., 'from my_app.utils import ...')
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)


def get_module_name_from_path(file_path: Path, project_root: Path) -> str:
    """
    Converts a file path to a Python module import path.
    e.g., /path/to/project/my_app/utils.py -> my_app.utils
    """
    relative_path = file_path.relative_to(project_root)

    # Remove .py extension
    module_path = relative_path.with_suffix('')

    # Handle __init__.py files (they represent the package itself)
    if module_path.name == '__init__':
        module_path = module_path.parent

    # Convert path separators to dots
    return str(module_path).replace(os.sep, '.')


def analyze_dependencies(start_dir: Path):
    """
    Analyzes all Python files in a directory and builds the dependency graph.
    """
    # { 'my_app.main': {'pandas', 'my_app.utils'}, ... }
    dependency_graph = defaultdict(set)

    # A set of all module names that are part of this project
    project_modules = set()

    # --- First Pass: Find all project modules ---
    py_files = []
    for root, dirs, files in os.walk(start_dir, topdown=True):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        current_dir = Path(root)
        for file in files:
            if file.endswith('.py'):
                file_path = current_dir / file
                py_files.append(file_path)
                module_name = get_module_name_from_path(file_path, start_dir)
                if module_name:  # Avoid empty string for top-level __init__.py
                    project_modules.add(module_name)

    # --- Second Pass: Parse files and find imports ---
    for file_path in py_files:
        current_module = get_module_name_from_path(file_path, start_dir)
        if not current_module:
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))

                visitor = ImportVisitor()
                visitor.visit(tree)

                # Add absolute imports
                dependency_graph[current_module].update(visitor.imports)

                # Resolve and add relative imports
                for rel_import in visitor.relative_imports:
                    resolved = resolve_relative_import(current_module, rel_import)
                    if resolved:
                        dependency_graph[current_module].add(resolved)

        except Exception as e:
            print(f"⚠️  Could not parse {file_path}: {e}")

    return dependency_graph, project_modules


def resolve_relative_import(current_module: str, relative_import: str) -> str:
    """
    Resolves a relative import path into an absolute project module path.
    e.g., current_module='my_app.views', relative_import='..models' -> 'my_app.models'
    """
    if not relative_import.startswith('.'):
        return relative_import  # Not a relative import

    # Get the "directory" of the current module
    # e.g., 'my_app.views.main' -> 'my_app.views'
    package_parts = current_module.split('.')[:-1]

    # Count the '..' levels
    level = 0
    for char in relative_import:
        if char == '.':
            level += 1
        else:
            break  # Stop at first non-dot character

    # e.g., '..models' -> level=2, remainder='models'
    remainder = relative_import[level:]

    if level > len(package_parts):
        # This is an invalid import (e.g., 'from ... import' in a top-level module)
        return f"(Invalid Relative Import: {relative_import})"

    # Go up 'level-1' directories
    base_parts = package_parts[:len(package_parts) - (level - 1)]

    if remainder:
        base_parts.append(remainder)

    return ".".join(base_parts)


def print_graph(dependency_graph, project_modules):
    """
    Prints a human-readable version of the dependency graph.
    """
    print(f"\n--- 🚀 Project Dependency Analysis ---")
    print(f"Found {len(project_modules)} local modules and analyzed {len(dependency_graph)} of them.")

    sorted_modules = sorted(list(dependency_graph.keys()))

    for module in sorted_modules:
        imports = dependency_graph.get(module, set())
        if not imports:
            continue

        print(f"\n📄 **{module}** is connected to:")

        internal_imports = {imp for imp in imports if imp in project_modules and imp != module}
        external_imports = {imp for imp in imports if imp not in project_modules}

        if internal_imports:
            print("  🔗 Internal (This Project):")
            for i_imp in sorted(list(internal_imports)):
                print(f"    - {i_imp}")

        if external_imports:
            print("  📦 External (Libraries):")
            for e_imp in sorted(list(external_imports)):
                print(f"    - {e_imp}")

        if not internal_imports and not external_imports:
            print("  (No imports found by parser)")


# --- How to run it ---
if __name__ == "__main__":
    # Use the current working directory as the project root
    PROJECT_DIRECTORY = Path.cwd()

    print(f"Analyzing Python files in: {PROJECT_DIRECTORY}\n")
    graph, modules = analyze_dependencies(PROJECT_DIRECTORY)
    print_graph(graph, modules)