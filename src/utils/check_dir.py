import os

def check(dir):
    if not os.path.exists(dir):
        print(f"Creating directory: {dir}")
        os.makedirs(dir)
    else:
        print(f"Directory already exists: {dir}")