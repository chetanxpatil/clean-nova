#!/usr/bin/env python3
"""
A dependency analyzer for Python projects.

Features:
- Scans all .py files in a directory
- Builds internal/external dependency graph
- Prints full graph OR a dependency tree from a specific file/module
- Use --all to expand bidirectional (imports + imported-by) connections
- Use --un to list modules not connected to a given file/module
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, Set, Tuple
from collections import defaultdict, deque

# Optional terminal enhancements
try:
    from colorama import Fore, Style
except ImportError:
    class Dummy:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
        RESET = RESET_ALL = ""
    Fore = Style = Dummy()

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x  # fallback if tqdm not installed

# Directories to skip
EXCLUDE_DIRS = {'venv', '.venv', '.git', '__pycache__', 'build', 'dist', '.vscode'}


# -------------------------------------------------------------------------
# AST Visitor
# -------------------------------------------------------------------------
class ImportVisitor(ast.NodeVisitor):
    """Extracts import statements from a Python file using the AST."""

    def __init__(self):
        self.imports: Set[str] = set()
        self.relative_imports: Set[str] = set()

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level > 0:
            prefix = "." * node.level
            module_name = prefix + (node.module or "")
            self.relative_imports.add(module_name)
        elif node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)


# -------------------------------------------------------------------------
# Core analysis functions
# -------------------------------------------------------------------------
def get_module_name_from_path(file_path: Path, project_root: Path) -> str:
    """Convert file path to Python module name relative to the project root."""
    relative = file_path.relative_to(project_root).with_suffix('')
    if relative.name == '__init__':
        relative = relative.parent
    return str(relative).replace(os.sep, '.')


def resolve_relative_import(current_module: str, relative_import: str, project_modules: Set[str] = None) -> str:
    """Resolve relative import path (e.g., '..utils') to absolute dotted path, verifying against project modules."""
    if not relative_import.startswith('.'):
        return relative_import

    parts = current_module.split('.')[:-1]
    level = len(relative_import) - len(relative_import.lstrip('.'))
    remainder = relative_import[level:] or ""

    if level > len(parts):
        return f"(Invalid Relative Import: {relative_import})"

    new_parts = parts[:-level] if level > 0 else parts
    if remainder:
        new_parts.append(remainder)
    candidate = ".".join(p for p in new_parts if p)

    # Verify resolved path exists in project
    if project_modules and candidate not in project_modules:
        # Try assuming it's a sibling within same package
        alt_candidate = f"{'.'.join(parts)}.{remainder}" if remainder else None
        if alt_candidate and alt_candidate in project_modules:
            return alt_candidate
        return f"(Invalid Relative Import: {relative_import})"
    return candidate


def analyze_dependencies(start_dir: Path) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Analyze Python files and return (dependency_graph, project_modules)."""
    dependency_graph: Dict[str, Set[str]] = defaultdict(set)
    project_modules: Set[str] = set()
    py_files = []

    # Pass 1: gather all project modules
    for root, dirs, files in os.walk(start_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith('.py'):
                path = Path(root) / file
                py_files.append(path)
                project_modules.add(get_module_name_from_path(path, start_dir))

    # Pass 2: parse imports
    for file_path in tqdm(py_files, desc="Analyzing files", ncols=80):
        module = get_module_name_from_path(file_path, start_dir)
        try:
            with open(file_path, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            visitor = ImportVisitor()
            visitor.visit(tree)

            deps = visitor.imports.copy()
            for rel in visitor.relative_imports:
                deps.add(resolve_relative_import(module, rel, project_modules))

            dependency_graph[module].update(deps)
        except SyntaxError as e:
            print(f"{Fore.RED}⚠️ Syntax error in {file_path}:{Style.RESET_ALL} {e}")
        except Exception as e:
            print(f"{Fore.RED}⚠️ Could not parse {file_path}:{Style.RESET_ALL} {e}")

    return dependency_graph, project_modules


# -------------------------------------------------------------------------
# Graph utilities
# -------------------------------------------------------------------------
def build_reverse_graph(graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Build reverse dependency graph (who imports whom)."""
    reverse = defaultdict(set)
    for mod, deps in graph.items():
        for dep in deps:
            reverse[dep].add(mod)
    return reverse


# -------------------------------------------------------------------------
# Tree printing
# -------------------------------------------------------------------------
def print_dependency_tree(graph: Dict[str, Set[str]],
                          start: str,
                          project_modules: Set[str],
                          reverse_graph=None,
                          include_reverse=False,
                          level: int = 0,
                          visited=None):
    """Recursively print dependencies in a tree format."""
    if visited is None:
        visited = set()
    indent = " " * (3 * level)
    prefix = "└── " if level > 0 else ""
    color = Fore.GREEN if start in project_modules else Fore.MAGENTA
    print(f"{indent}{prefix}{color}{start}{Style.RESET_ALL}")

    if start in visited:
        print(f"{indent}   ↩︎ (already visited)")
        return
    visited.add(start)

    neighbors = set(graph.get(start, []))
    if include_reverse and reverse_graph:
        neighbors |= reverse_graph.get(start, set())

    for dep in sorted(neighbors):
        print_dependency_tree(graph, dep, project_modules,
                              reverse_graph=reverse_graph,
                              include_reverse=include_reverse,
                              level=level + 1,
                              visited=visited)


# -------------------------------------------------------------------------
# Output functions
# -------------------------------------------------------------------------
def print_graph(graph: Dict[str, Set[str]], project_modules: Set[str]) -> None:
    """Pretty-print full dependency graph."""
    print(f"\n{Fore.CYAN}--- 🚀 Project Dependency Analysis ---{Style.RESET_ALL}")
    print(f"Found {len(project_modules)} local modules and analyzed {len(graph)} of them.\n")

    for module in sorted(graph.keys()):
        imports = graph[module]
        if not imports:
            continue

        print(f"{Fore.YELLOW}📄 {module}{Style.RESET_ALL} is connected to:")
        internal = sorted(i for i in imports if i in project_modules and i != module)
        external = sorted(i for i in imports if i not in project_modules)

        if internal:
            print(f"  {Fore.GREEN}🔗 Internal:{Style.RESET_ALL}")
            for imp in internal:
                print(f"    - {imp}")

        if external:
            print(f"  {Fore.MAGENTA}📦 External:{Style.RESET_ALL}")
            for imp in external:
                print(f"    - {imp}")
        print("")


# -------------------------------------------------------------------------
# Helper: find unconnected modules
# -------------------------------------------------------------------------
def find_unconnected_modules(graph: Dict[str, Set[str]], start_module: str) -> Set[str]:
    """Return all modules that are NOT connected (in either direction) to start_module."""
    reverse = build_reverse_graph(graph)
    visited = set()
    queue = deque([start_module])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        neighbors = graph.get(node, set()) | reverse.get(node, set())
        for n in neighbors:
            if n not in visited:
                queue.append(n)

    all_modules = set(graph.keys())
    return all_modules - visited


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path.cwd()
    args = sys.argv[1:]

    if not args:
        target_module = None
        show_all = False
        show_unconnected = False
    else:
        target_module = next((a for a in args if not a.startswith("--")), None)
        show_all = "--all" in args
        show_unconnected = "--un" in args

    print(f"Analyzing Python files in: {project_root}\n")
    graph, modules = analyze_dependencies(project_root)
    reverse_graph = build_reverse_graph(graph)

    if target_module:
        # Normalize argument: file path or module name
        arg_path = Path(target_module)
        if arg_path.exists() and arg_path.suffix == '.py':
            target_module = get_module_name_from_path(arg_path.resolve(), project_root)
        else:
            target_module = target_module.replace('/', '.').replace('\\', '.').removesuffix('.py')

        if target_module not in graph and target_module not in modules:
            print(f"{Fore.RED}❌ Module '{target_module}' not found in project.{Style.RESET_ALL}")
            sys.exit(1)

        print(f"\n{Fore.CYAN}--- 🌳 Dependency Tree for {target_module} ---{Style.RESET_ALL}\n")
        print_dependency_tree(graph, target_module, modules,
                              reverse_graph=reverse_graph,
                              include_reverse=show_all)

        if show_unconnected:
            unconnected = find_unconnected_modules(graph, target_module)
            if unconnected:
                percent = 100 * (1 - len(unconnected) / len(graph))
                print(f"\n{Fore.YELLOW}--- 🧩 Modules NOT connected to {target_module} ---{Style.RESET_ALL}")
                for mod in sorted(unconnected):
                    print(f"  - {mod}")
                print(f"\n{Fore.CYAN}📊 Connectivity: {percent:.1f}% of modules are connected to {target_module}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.GREEN}✅ All modules are connected to {target_module}!{Style.RESET_ALL}")

    else:
        print(f"{Fore.YELLOW}No file specified. Use a file or module name, e.g.:{Style.RESET_ALL}")
        print("  python analyze.py growth/main.py")
        print("  python analyze.py growth/main.py --all")
        print("  python analyze.py growth/main.py --un")
