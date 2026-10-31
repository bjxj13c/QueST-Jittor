"""Read-only loading of pinned upstream tokenizer for numerical comparisons."""
import importlib.util
from pathlib import Path
import subprocess
import ast
import types

from .config import TokenizerConfig

QUEST_COMMIT = "cc25a835a00cf2d91df69704bdd757904d13c061"


def load_upstream_module(root):
    repository = Path(root) / "upstream/QueST"
    actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if actual != QUEST_COMMIT:
        raise RuntimeError(f"Unexpected upstream revision: {actual}")
    filename = repository / "quest/algos/quest_modules/skill_vae.py"
    spec = importlib.util.spec_from_file_location("quest_original_skill_vae", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_upstream_tokenizer(root, config=None):
    config = config or TokenizerConfig()
    arguments = config.to_dict()
    arguments["fsq_level"] = list(arguments.pop("levels"))
    arguments.update(vq_type="fsq", codebook_dim=512, codebook_size=1024)
    return load_upstream_module(root).SkillVAE(**arguments)


def load_selected_definitions(root, relative_path, names, namespace):
    """Execute exact upstream AST definitions, skipping unrelated optional imports.

    Classes/functions themselves are not rewritten. This allows targeted parity
    checks without installing unrelated LIBERO/ACT/robomimic dependencies.
    """
    repository = Path(root) / "upstream/QueST"
    actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if actual != QUEST_COMMIT:
        raise RuntimeError("Unexpected upstream revision")
    filename = repository / relative_path
    parsed = ast.parse(filename.read_text(encoding="utf-8"))
    nodes = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if set(node.name for node in nodes) != set(names):
        raise ValueError("Missing upstream definitions")
    module = types.ModuleType("quest_selected_reference")
    module.__dict__.update(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), "exec"), module.__dict__)
    return module
