#!/usr/bin/env python3
"""
Tree-sitter Erlang parser setup for Erlang corpus scraper.
Handles Erlang AST parsing and basic tree navigation.
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from tree_sitter import Language, Parser, Node

# Setup logging
logger = logging.getLogger(__name__)

class ErlangParser:
    """Tree-sitter based Erlang parser."""

    def __init__(self, parser_dir: str = "parsers"):
        """Initialize Erlang parser.

        Args:
            parser_dir: Directory to store parser files
        """
        self.parser_dir = Path(parser_dir)
        self.parser_dir.mkdir(exist_ok=True)

        self.language = None
        self.parser = None
        self._setup_parser()

    def _setup_parser(self):
        """Set up tree-sitter Erlang parser."""
        try:
            # Check if language library exists
            lib_path = self.parser_dir / "erlang.so"

            if not lib_path.exists():
                logger.info("Building Erlang tree-sitter library...")
                self._build_erlang_library()

            # Load language and create parser
            self._load_language(lib_path)

            logger.info("✓ Erlang parser initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Erlang parser: {e}")
            raise

    def _load_language(self, lib_path: Path):
        """Load the language using the appropriate API."""
        import ctypes

        # Try different methods to load the language
        methods = [
            self._try_new_api_with_ctypes,
            self._try_old_api_two_args,
            self._try_old_api_with_set_language
        ]

        for i, method in enumerate(methods, 1):
            try:
                method(lib_path)
                logger.info(f"✓ Successfully loaded language using method {i}")
                return
            except Exception as e:
                logger.debug(f"Method {i} failed: {e}")
                continue

        raise Exception("All language loading methods failed")

    def _try_new_api_with_ctypes(self, lib_path: Path):
        """Try new API with ctypes loading."""
        import ctypes

        # Load the shared library
        lib = ctypes.cdll.LoadLibrary(str(lib_path))

        # Debug: List all available symbols
        self._debug_library_symbols(lib_path)

        # Try common function names for tree-sitter languages
        function_names = ['tree_sitter_erlang', 'tree_sitter_erl', 'language']

        for func_name in function_names:
            try:
                logger.debug(f"Trying to get function: {func_name}")

                # Method 1: Direct getattr
                try:
                    language_func = getattr(lib, func_name)
                    logger.debug(f"✓ Found function {func_name} with getattr")
                except AttributeError:
                    logger.debug(f"✗ getattr failed for {func_name}")

                    # Method 2: Try with ctypes.CDLL explicit loading
                    try:
                        lib2 = ctypes.CDLL(str(lib_path))
                        language_func = getattr(lib2, func_name)
                        logger.debug(f"✓ Found function {func_name} with CDLL")
                        lib = lib2  # Use this library instance
                    except AttributeError:
                        logger.debug(f"✗ CDLL also failed for {func_name}")

                        # Method 3: Try manual symbol lookup
                        try:
                            language_func = lib[func_name]
                            logger.debug(f"✓ Found function {func_name} with [] notation")
                        except (KeyError, AttributeError):
                            logger.debug(f"✗ Manual lookup failed for {func_name}")
                            continue

                # Configure function signature
                language_func.restype = ctypes.c_void_p
                language_func.argtypes = []

                # Test calling the function
                try:
                    result = language_func()
                    logger.debug(f"✓ Successfully called {func_name}(), result: {result}")

                    if result is None or result == 0:
                        logger.warning(f"Function {func_name}() returned null/zero")
                        continue

                except Exception as e:
                    logger.error(f"✗ Failed to call {func_name}(): {e}")
                    continue

                # Try to create Language object
                try:
                    self.language = Language(result)
                    logger.info(f"✓ Successfully created Language object")
                except Exception as e:
                    logger.error(f"✗ Failed to create Language object: {e}")
                    continue

                # Try to create Parser object
                try:
                    # Check Parser constructor signature
                    import inspect
                    parser_sig = inspect.signature(Parser.__init__)
                    logger.info(f"Parser.__init__ signature: {parser_sig}")

                    if len(parser_sig.parameters) > 1:  # More than just 'self'
                        self.parser = Parser(self.language)
                        logger.info(f"✓ Created Parser with language argument")
                    else:
                        self.parser = Parser()
                        if hasattr(self.parser, 'set_language'):
                            self.parser.set_language(self.language)
                            logger.info(f"✓ Created Parser and set language separately")
                        else:
                            raise Exception("Parser has no set_language method")

                except Exception as e:
                    logger.error(f"✗ Failed to create Parser: {e}")
                    continue

                return  # Success!

            except Exception as e:
                logger.error(f"✗ Overall failure for {func_name}: {e}")
                continue

        raise Exception(f"No valid language function found in {lib_path}")

    def _debug_library_symbols(self, lib_path: Path):
        """Debug: Try to list symbols in the library."""
        try:
            # Try using objdump if available
            result = subprocess.run([
                'objdump', '-T', str(lib_path)
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                logger.debug("Available symbols in library:")
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'tree_sitter' in line.lower() or 'erlang' in line.lower():
                        logger.debug(f"  {line.strip()}")
            else:
                logger.debug("objdump failed, trying nm...")

                # Try nm as alternative
                result = subprocess.run([
                    'nm', '-D', str(lib_path)
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'tree_sitter' in line.lower() or 'erlang' in line.lower():
                            logger.debug(f"  {line.strip()}")

        except Exception as e:
            logger.info(f"Could not debug symbols: {e}")

        # Also try to list what ctypes can see
        try:
            import ctypes
            lib = ctypes.cdll.LoadLibrary(str(lib_path))
            logger.info(f"Library loaded successfully: {lib}")

            # Try a few common symbols that should exist
            test_symbols = ['tree_sitter_erlang', '_tree_sitter_erlang', 'tree_sitter_erl']
            for symbol in test_symbols:
                try:
                    func = getattr(lib, symbol)
                    logger.debug(f"✓ ctypes can access: {symbol}")
                except AttributeError:
                    logger.debug(f"✗ ctypes cannot access: {symbol}")

        except Exception as e:
            logger.info(f"ctypes debug failed: {e}")

    def _try_old_api_two_args(self, lib_path: Path):
        """Try old API with two arguments."""
        self.language = Language(str(lib_path), 'erlang')
        self.parser = Parser()
        self.parser.set_language(self.language)

    def _try_old_api_with_set_language(self, lib_path: Path):
        """Try another variant of old API."""
        self.language = Language(str(lib_path))
        self.parser = Parser()
        self.parser.set_language(self.language)

    def _build_erlang_library(self):
        """Build the tree-sitter Erlang language library."""
        # Clone tree-sitter-erlang if needed
        erlang_repo_path = self.parser_dir / "tree-sitter-erlang"

        if not erlang_repo_path.exists():
            logger.info("Cloning tree-sitter-erlang repository...")
            try:
                subprocess.run([
                    "git", "clone",
                    "https://github.com/WhatsApp/tree-sitter-erlang.git",
                    str(erlang_repo_path)
                ], check=True, capture_output=True, text=True)
                logger.info("✓ Successfully cloned tree-sitter-erlang")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone tree-sitter-erlang: {e}")
                raise

        # Build language library using tree-sitter CLI
        lib_path = self.parser_dir / "erlang.so"
        try:
            # Method 1: Try using tree-sitter CLI if available
            try:
                subprocess.run([
                    "tree-sitter", "build",
                    "--output", str(lib_path),
                    str(erlang_repo_path)
                ], check=True, capture_output=True, text=True, cwd=str(erlang_repo_path))
                logger.info("✓ Successfully built Erlang language library using tree-sitter CLI")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info("tree-sitter CLI not available, trying alternative method...")

            # Method 2: Try the old build_library method (for older py-tree-sitter versions)
            try:
                if hasattr(Language, 'build_library'):
                    Language.build_library(
                        str(lib_path),
                        [str(erlang_repo_path)]
                    )
                    logger.info("✓ Successfully built Erlang language library using build_library")
                    return
            except Exception as e:
                logger.info(f"build_library method failed: {e}")

            # Method 3: Manual compilation using subprocess
            logger.info("Attempting manual compilation...")
            self._manual_compile_erlang(erlang_repo_path, lib_path)

        except Exception as e:
            logger.error(f"Failed to build Erlang language library: {e}")
            raise

    def _manual_compile_erlang(self, repo_path: Path, output_path: Path):
        """Manually compile tree-sitter-erlang using gcc/clang."""
        import platform

        # Determine compiler and flags
        if platform.system() == "Windows":
            # For Windows, we'd need different approach, but let's focus on Unix first
            raise Exception("Windows compilation not supported yet")

        # Unix/Linux/MacOS compilation
        c_files = list(repo_path.glob("src/*.c"))
        if not c_files:
            raise Exception(f"No C source files found in {repo_path}/src/")

        # Basic compilation command
        cmd = [
            "gcc" if shutil.which("gcc") else "clang",
            "-shared", "-fPIC", "-O2",
            "-I", str(repo_path / "src"),
            "-o", str(output_path)
        ] + [str(f) for f in c_files]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✓ Successfully compiled Erlang language library manually")
        except subprocess.CalledProcessError as e:
            logger.error(f"Manual compilation failed: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise

    # === BASIC PARSER METHODS ===

    def parse_string(self, code: str) -> Optional[Node]:
        """Parse Erlang code string into AST."""
        if not self.parser:
            logger.error("Parser not initialized")
            return None

        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree.root_node
        except Exception as e:
            logger.error(f"Failed to parse Erlang code: {e}")
            return None

    def parse_file(self, file_path: str) -> Optional[Node]:
        """Parse Erlang file into AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return self.parse_string(code)
        except Exception as e:
            logger.error(f"Failed to read/parse file {file_path}: {e}")
            return None

    def node_text(self, node: Node) -> str:
        """Get text content of a node."""
        return node.text.decode('utf-8') if node.text else ""

    # === CLEAN FUNCTION EXTRACTION ===

    def extract_functions(self, root_node: Node) -> List[Node]:
        """Extract function definitions, grouping multi-clause functions."""
        # Step 1: Find all function declarations
        fun_decls = []

        def find_fun_decls(node):
            if node.type == 'fun_decl':
                fun_decls.append(node)
            for child in node.children:
                find_fun_decls(child)

        find_fun_decls(root_node)
        logger.debug(f"Found {len(fun_decls)} fun_decl nodes")

        # Step 2: Group by function name/arity
        function_groups = {}

        for fun_decl in fun_decls:
            name = self.get_function_name(fun_decl)
            arity = self.get_function_arity(fun_decl)

            if name is None:
                continue

            sig = f"{name}/{arity}"
            if sig not in function_groups:
                function_groups[sig] = []
            function_groups[sig].append(fun_decl)

        logger.debug(f"Grouped into: {list(function_groups.keys())}")

        # Step 3: Create combined functions for multi-clause functions
        result = []
        for sig, clauses in function_groups.items():
            if len(clauses) == 1:
                result.append(clauses[0])
            else:
                combined = self._create_combined_function(clauses, sig)
                if combined:
                    result.append(combined)

        return result

    def _create_combined_function(self, clauses: List[Node], sig: str):
        """Create a combined function node from multiple clauses."""
        class CombinedFunction:
            def __init__(self, clauses, sig):
                self.clauses = clauses
                self.sig = sig
                self.type = 'combined_function'

                # Use span of all clauses
                self.start_point = clauses[0].start_point
                self.end_point = clauses[-1].end_point

                # Combine text with semicolons
                texts = []
                for clause in clauses:
                    text = clause.text.decode('utf-8').strip()
                    if text.endswith(';'):
                        text = text[:-1]
                    elif text.endswith('.'):
                        text = text[:-1]
                    texts.append(text)

                self._combined_text = '; '.join(texts) + '.'
                self.children = clauses
                self.parent = clauses[0].parent if clauses else None

            @property
            def text(self):
                return self._combined_text.encode('utf-8')

        return CombinedFunction(clauses, sig)

    def get_function_name(self, func_node: Node) -> Optional[str]:
        """Get function name from any function node."""
        if hasattr(func_node, 'type') and func_node.type == 'combined_function':
            return self._extract_name_from_fun_decl(func_node.clauses[0])
        else:
            return self._extract_name_from_fun_decl(func_node)

    def _extract_name_from_fun_decl(self, fun_decl: Node) -> Optional[str]:
        """Extract function name from fun_decl node."""
        # Look for function_clause -> atom
        for child in fun_decl.children:
            if child.type == 'function_clause':
                for grandchild in child.children:
                    if grandchild.type == 'atom':
                        return self.node_text(grandchild)
        return None

    def get_function_arity(self, func_node: Node) -> int:
        """Get function arity from any function node."""
        if hasattr(func_node, 'type') and func_node.type == 'combined_function':
            return self._extract_arity_from_fun_decl(func_node.clauses[0])
        else:
            return self._extract_arity_from_fun_decl(func_node)

    def _extract_arity_from_fun_decl(self, fun_decl: Node) -> int:
        """Extract function arity from fun_decl node."""
        # Look for function_clause -> expr_args and count parameters
        for child in fun_decl.children:
            if child.type == 'function_clause':
                for grandchild in child.children:
                    if grandchild.type == 'expr_args':
                        return self._count_parameters_in_expr_args(grandchild)
        return 0

    def _count_parameters_in_expr_args(self, expr_args: Node) -> int:
        """Count parameters in expr_args node."""
        param_count = 0
        for child in expr_args.children:
            # Count actual parameter nodes, skip punctuation
            if child.type in ['var', 'atom', 'integer', 'float', 'string', 'binary', 'list', 'tuple']:
                param_count += 1
        return param_count

    def get_function_clauses(self, func_node: Node) -> List[Node]:
        """Get all clauses of a function."""
        if hasattr(func_node, 'type') and func_node.type == 'combined_function':
            # For combined functions, extract function_clause from each fun_decl
            clauses = []
            for fun_decl in func_node.clauses:
                for child in fun_decl.children:
                    if child.type == 'function_clause':
                        clauses.append(child)
            return clauses
        else:
            # For single fun_decl, find its function_clause
            for child in func_node.children:
                if child.type == 'function_clause':
                    return [child]
            return []

    def has_guard(self, clause_node: Node) -> bool:
        """Check if a function clause has a guard."""
        for child in clause_node.children:
            if child.type in ['when', 'guard']:
                return True
        return False

    # === GRAPHCODEBERT DATA EXTRACTION ===

    def extract_graphcodebert_data(self, func_node: Node, file_lines: List[str]) -> Tuple[List[str], List[int], List[str], List[bool]]:
        """Extract GraphCodeBERT data with clause parameter information.
        
        Returns:
            tokens: List of code tokens
            var_indices: List of variable token positions
            var_names: List of variable names
            is_clause_params: List of booleans indicating if each variable is a clause parameter
        """
        if hasattr(func_node, 'type') and func_node.type == 'combined_function':
            # Handle combined functions
            all_tokens = []
            all_var_indices = []
            all_var_names = []
            all_is_clause_params = []
    
            for fun_decl in func_node.clauses:
                tokens, var_indices, var_names, is_clause_params = self._extract_tokens_and_variables(fun_decl)
    
                # Adjust variable indices for combined sequence
                offset = len(all_tokens)
                adjusted_indices = [idx + offset for idx in var_indices]
    
                all_tokens.extend(tokens)
                all_var_indices.extend(adjusted_indices)
                all_var_names.extend(var_names)
                all_is_clause_params.extend(is_clause_params)
    
            return all_tokens, all_var_indices, all_var_names, all_is_clause_params
        else:
            # Handle single function
            return self._extract_tokens_and_variables(func_node)
    
    def _extract_tokens_and_variables(self, node: Node) -> Tuple[List[str], List[int], List[str], List[bool]]:
        """Extract tokens and identify variables with clause parameter tracking."""
        tokens = []
        token_nodes = []
    
        # Collect all leaf tokens
        def collect_tokens(n):
            if len(n.children) == 0 and n.type != 'comment':
                text = self.node_text(n).strip()
                if text:
                    tokens.append(text)
                    token_nodes.append(n)
            else:
                for child in n.children:
                    collect_tokens(child)
    
        collect_tokens(node)
    
        # Get clause parameter positions by analyzing the AST structure
        clause_param_positions = self._identify_clause_parameters(node)
    
        # Identify variables and mark clause parameters
        var_indices = []
        var_names = []
        is_clause_params = []
    
        for i, (token, token_node) in enumerate(zip(tokens, token_nodes)):
            # Erlang variables: start with uppercase or underscore, or marked as 'var'
            is_variable = (
                token_node.type == 'var' or
                (token and len(token) > 0 and (token[0].isupper() or token[0] == '_'))
            )
    
            if is_variable:
                var_indices.append(i)
                var_names.append(token)
                
                # Check if this variable is a clause parameter
                is_param = self._is_token_clause_parameter(token_node, clause_param_positions)
                is_clause_params.append(is_param)
    
        return tokens, var_indices, var_names, is_clause_params
    
    def _identify_clause_parameters(self, func_node: Node) -> set:
        """Identify the AST positions of clause parameters.
        
        Returns a set of (start_byte, end_byte) tuples for parameter nodes.
        """
        param_positions = set()
        
        # Find function_clause nodes
        def find_clauses(node):
            if node.type == 'function_clause':
                # Look for expr_args (parameter list)
                for child in node.children:
                    if child.type == 'expr_args':
                        # Collect all variable nodes within expr_args
                        self._collect_variable_positions(child, param_positions)
            else:
                for child in node.children:
                    find_clauses(child)
        
        find_clauses(func_node)
        return param_positions
    
    def _collect_variable_positions(self, node: Node, positions: set):
        """Recursively collect positions of variable nodes."""
        if node.type == 'var':
            positions.add((node.start_byte, node.end_byte))
        else:
            for child in node.children:
                self._collect_variable_positions(child, positions)
    
    def _is_token_clause_parameter(self, token_node: Node, clause_param_positions: set) -> bool:
        """Check if a token node is a clause parameter based on its position."""
        if not hasattr(token_node, 'start_byte') or not hasattr(token_node, 'end_byte'):
            return False
        
        token_pos = (token_node.start_byte, token_node.end_byte)
        return token_pos in clause_param_positions
    
    def create_dataflow_graph(self, var_indices: List[int], var_names: List[str], is_clause_params: List[bool] = None) -> List[Tuple[str, int, str, List[str], List[int]]]:
        """Create dataflow graph with simplified semantics when clause parameter info is available.
        
        Args:
            var_indices: Variable token positions
            var_names: Variable names
            is_clause_params: Optional list indicating which variables are clause parameters.
                             If None, falls back to old behavior for backward compatibility.
        
        Returns:
            DFG edges in format: (var_name, position, relation, dependencies, dependency_positions)
        """
        dfg = []
        var_states = {}  # Track variable usage
    
        if is_clause_params is None:
            # Backward compatibility: old behavior without clause parameter info
            for idx, name in zip(var_indices, var_names):
                if name in var_states:
                    # Variable used before - create dependency on all previous occurrences
                    prev_indices = var_states[name].copy()
                    dfg.append((name, idx, 'comesFrom', [name], prev_indices))
                    var_states[name].append(idx)
                else:
                    # First occurrence
                    dfg.append((name, idx, 'comesFrom', [], []))
                    var_states[name] = [idx]
        else:
            # New behavior with simplified semantics using clause parameter info
            for idx, name, is_param in zip(var_indices, var_names, is_clause_params):
                if is_param:
                    # Clause parameter - independent variable
                    dfg.append((name, idx, 'comesFrom', [], []))
                    var_states[name] = [idx]
                else:
                    # Non-parameter variable - check for data flow
                    if name in var_states:
                        # Variable used before - link to most recent occurrence
                        prev_positions = var_states[name]
                        most_recent = prev_positions[-1]  # Get the actual source position
                        dfg.append((name, idx, 'comesFrom', [name], [most_recent]))
                        var_states[name].append(idx)
                    else:
                        # First occurrence of this variable (not a parameter)
                        dfg.append((name, idx, 'comesFrom', [], []))
                        var_states[name] = [idx]
    
        return dfg

    def print_ast(self, node: Node, indent: int = 0) -> None:
        """Print AST structure for debugging."""
        indent_str = "  " * indent
        text = self.node_text(node)[:50].replace('\n', '\\n')
        print(f"{indent_str}{node.type}: '{text}'")

        for child in node.children:
            self.print_ast(child, indent + 1)


def test_parser():
    """Test the hybrid parser."""
    logging.basicConfig(level=logging.INFO)

    test_code = '''
    %% Test function with multiple clauses
    -spec max(number(), number()) -> number().
    max(A, B) when A > B -> A;
    max(A, B) -> B.

    %% Calculate factorial
    factorial(0) -> 1;
    factorial(N) when N > 0 -> N * factorial(N - 1).

    %% Simple function
    helper() -> ok.
    '''

    try:
        parser = ErlangParser()
        root = parser.parse_string(test_code)

        if root:
            print("✓ Parse successful")

            # Extract functions
            functions = parser.extract_functions(root)
            print(f"Found {len(functions)} functions:")

            for func in functions:
                name = parser.get_function_name(func)
                arity = parser.get_function_arity(func)
                clauses = parser.get_function_clauses(func)

                print(f"  {name}/{arity} - {len(clauses)} clause(s)")

                # Extract GraphCodeBERT data
                tokens, var_indices, var_names = parser.extract_graphcodebert_data(func, test_code.split('\n'))
                dfg = parser.create_dataflow_graph(var_indices, var_names)

                print(f"    Tokens: {tokens}")
                print(f"    Variables: {list(zip(var_indices, var_names))}")
                print(f"    DFG: {dfg}")
                print()

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_parser()
