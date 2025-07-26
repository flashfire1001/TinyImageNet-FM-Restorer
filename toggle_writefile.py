import argparse
import nbformat
import sys

def comment_writefile(nb):
    """Replace '%%writefile' with '#%%writefile' in code cells."""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            new_lines = []
            for line in cell.source.splitlines():
                if line.strip().startswith('%%writefile'):
                    new_lines.append(line.replace('%%', '#%%', 1))
                else:
                    new_lines.append(line)
            cell.source = '\n'.join(new_lines)
    return nb

def revert_writefile(nb):
    """Replace '#%%writefile' with '%%writefile' in code cells."""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            new_lines = []
            for line in cell.source.splitlines():
                if line.strip().startswith('#%%writefile'):
                    new_lines.append(line.replace('#%%', '%%', 1))
                else:
                    new_lines.append(line)
            cell.source = '\n'.join(new_lines)
    return nb

def main():
    parser = argparse.ArgumentParser(description="Toggle %%writefile comments in a .ipynb file.")
    parser.add_argument("file", help="Path to the input .ipynb file")
    parser.add_argument("-c", "--comment", action="store_true", help="Comment out %%writefile lines")
    parser.add_argument("-r", "--revert", action="store_true", help="Revert #%%writefile lines back to %%writefile")

    args = parser.parse_args()

    if not args.comment and not args.revert:
        print("N Please specify either -c (comment) or -r (revert).")
        sys.exit(1)

    # Load the notebook
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"N Failed to read notebook: {e}")
        sys.exit(1)

    # Process
    if args.comment:
        nb = comment_writefile(nb)
    elif args.revert:
        nb = revert_writefile(nb)

    # Save
    try:
        with open(args.file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Y Notebook updated: {args.file}")
    except Exception as e:
        print(f"N Failed to write notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
