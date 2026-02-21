import os
import re


def bump_patch_version(version_str):
    major, minor, patch = map(int, version_str.split("."))
    return f"{major}.{minor}.{patch + 1}"


def update_file(filepath, search_str, replacement_str):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    with open(filepath) as f:
        content = f.read()
    if search_str not in content:
        print(f"Search string not found in {filepath}: {search_str}")
        return
    new_content = content.replace(search_str, replacement_str)
    with open(filepath, "w") as f:
        f.write(new_content)
    print(f"Updated {filepath}")


def main():
    # 1. Read current version from pyproject.toml
    with open("pyproject.toml") as f:
        pyproject_content = f.read()

    version_match = re.search(r'version\s*=\s*"([^"]+)"', pyproject_content)
    if not version_match:
        print("Could not find version in pyproject.toml")
        return

    current_version = version_match.group(1)
    new_version = bump_patch_version(current_version)
    print(f"Bumping version: {current_version} -> {new_version}")

    # 2. Update pyproject.toml
    update_file("pyproject.toml", f'version = "{current_version}"', f'version = "{new_version}"')

    # 3. Update README.md
    # Update badge
    update_file("README.md", f"version-{current_version}-blue", f"version-{new_version}-blue")
    # Update version text at bottom
    update_file("README.md", f"**Version**: {current_version}", f"**Version**: {new_version}")

    # 4. Update docs/en/index.md
    update_file("docs/en/index.md", f"**Version**: {current_version}", f"**Version**: {new_version}")

    # 5. Update docs/en/getting-started.md
    update_file("docs/en/getting-started.md", f"**Version**: {current_version}", f"**Version**: {new_version}")

    # Output the new version for GitHub Actions
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"new_version={new_version}\n")


if __name__ == "__main__":
    main()
