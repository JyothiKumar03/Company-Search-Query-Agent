"""
Interactive terminal entry point for the Company Search Query Agent.
Run: python cli.py
"""

from main import run


BANNER = """
╔══════════════════════════════════════════════════════════╗
║          Company Search Query Agent                      ║
║  Search 20M+ businesses using natural language           ║
║  Type 'quit' to exit                                     ║
╚══════════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)

    while True:
        try:
            query = input("Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        run(query)
        print()


if __name__ == "__main__":
    main()
