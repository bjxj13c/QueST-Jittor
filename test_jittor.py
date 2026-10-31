"""Run closed-loop evaluation with an existing native Jittor checkpoint."""
from scripts.entrypoints import main

if __name__ == "__main__":
    main("jittor", training=False)
