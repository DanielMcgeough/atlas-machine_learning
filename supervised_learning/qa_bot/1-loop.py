#!/usr/bin/env python3
"""
This script takes user input, responds with "A: ", and exits on specific input.
"""

if __name__ == '__main__':
    while True:
        user_input = input("Q: ")
        if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        else:
            print(f"A: {user_input}")
