# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2026 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
import os
import argparse

def create_project_structure(project_name, destination):
    project_structure = {
        f"{project_name}-checkpoints": [],
        f"{project_name}-files": [],
        f"{project_name}-imgs": [],
        f"{project_name}-results": ["analysis", "figures", "metrics", "models", "patients"]}

    def build_dir(base_path, structure):
        if isinstance(structure, list):
            for folder in structure:
                os.makedirs(os.path.join(base_path, folder), exist_ok=True)
        elif isinstance(structure, dict):
            for parent, children in structure.items():
                parent_path = os.path.join(base_path, parent)
                os.makedirs(parent_path, exist_ok=True)
                build_dir(parent_path, children)

    os.makedirs(destination, exist_ok=True)
    build_dir(destination, project_structure)

    print(f"Project '{project_name}' structure created successfully at: {destination}")

def main():
    parser = argparse.ArgumentParser(description="Create a project folder structure")
    parser.add_argument('--project_name', type=str, required=True, help='Name of the project prefix')
    parser.add_argument('--destination', type=str, required=True, help='Main path where folders will be created')
    
    args = parser.parse_args()
    create_project_structure(args.project_name, args.destination)

if __name__ == "__main__":
    main()