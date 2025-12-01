import os
import pyperclip

# Folders to ignore (relative to EACH root folder in folder_paths)
IGNORE_DIRS = [
    r".git",
    r"__pycache__",
    r".venv",
    r"venv",
    r"node_modules",
    r"build",
    r"dist",
    # examples:
    # r"data",
    # r"logs",
    # r"outputs\checkpoints",
]

# (Optional) ignore specific file extensions
IGNORE_FILE_EXTS = {
    # ".pt", ".pth", ".pkl", ".zip", ".png"
}

# (Optional) limit insane file sizes (bytes)
MAX_FILE_BYTES = 2_000_000  # 2MB


def _norm_relpath(relpath: str) -> str:
    """Normalize for consistent comparisons across platforms."""
    relpath = relpath.strip().strip("/\\")
    return os.path.normcase(os.path.normpath(relpath))


def _should_ignore_dir(root_folder: str, dir_path: str, ignore_rel_dirs_norm: set[str]) -> bool:
    rel = os.path.relpath(dir_path, root_folder)
    rel_norm = _norm_relpath(rel)
    # Ignore if the dir itself is (or is under) any ignored rel path
    for ign in ignore_rel_dirs_norm:
        if rel_norm == ign or rel_norm.startswith(ign + os.sep):
            return True
    return False


def generate_folder_tree(root_folder: str, current_folder: str, ignore_rel_dirs_norm: set[str]):
    """
    Recursively generates the folder tree structure with names and files,
    skipping ignored directories (relative to root_folder).
    Returns: list of tuples ("Folder"/"File", display_path, content_or_None)
    """
    tree_structure = []

    try:
        items = sorted(os.listdir(current_folder))
    except FileNotFoundError:
        print("Error: The folder path does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for accessing {current_folder}")
        return []

    for item in items:
        item_path = os.path.join(current_folder, item)

        if os.path.isdir(item_path):
            if _should_ignore_dir(root_folder, item_path, ignore_rel_dirs_norm):
                continue

            rel_display = os.path.relpath(item_path, root_folder)
            tree_structure.append(("Folder", rel_display, None))
            tree_structure.extend(generate_folder_tree(root_folder, item_path, ignore_rel_dirs_norm))

        elif os.path.isfile(item_path):
            ext = os.path.splitext(item)[1].lower()
            if ext in IGNORE_FILE_EXTS:
                continue

            rel_display = os.path.relpath(item_path, root_folder)

            try:
                size = os.path.getsize(item_path)
                if MAX_FILE_BYTES is not None and size > MAX_FILE_BYTES:
                    tree_structure.append(("File", rel_display, f"[Skipped: file too large ({size} bytes)]"))
                    continue

                with open(item_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                tree_structure.append(("File", rel_display, content))
            except Exception as e:
                tree_structure.append(("File", rel_display, f"Error reading file: {e}"))
        else:
            rel_display = os.path.relpath(item_path, root_folder)
            tree_structure.append(("Unknown", rel_display, None))

    return tree_structure


def display_folder_tree(tree_structure):
    output = []
    for item_type, name, content in tree_structure:
        if item_type == "Folder":
            output.append(f"[Folder] {name}")
        elif item_type == "File":
            output.append(f"[File] {name}")
            output.append("    Content:")
            output.append("    " + (content or "").replace("\n", "\n    "))
        else:
            output.append(f"[Unknown] {name}")
    return "\n".join(output)


if __name__ == "__main__":
    folder_paths = [r"C:\Users\fridm\Desktop\GraphPropertyPredict"]

    # Normalize ignore list once (relative paths)
    ignore_rel_dirs_norm = {_norm_relpath(p) for p in IGNORE_DIRS}

    tree_output = []
    for folder_path in folder_paths:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"Generating Folder Tree and Contents for:\n  {folder_path}\n")

            root_name = os.path.basename(os.path.normpath(folder_path))
            tree_output.append(f"=== ROOT: {root_name} ({folder_path}) ===")

            tree_structure = generate_folder_tree(folder_path, folder_path, ignore_rel_dirs_norm)
            tree_output.append(display_folder_tree(tree_structure))
        else:
            print(f"Invalid folder path: {folder_path}")

    final_output = "\n\n".join(tree_output)
    pyperclip.copy(final_output)
    print("The folder tree and contents have been copied to the clipboard.")
