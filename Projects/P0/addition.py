# addition.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
Run python autograder.py
"""

def add(a, b):
    "Return the sum of a and b"
    # <note>: Looking at addition[1,2,3].test, the types seem to be just ints|floats
    # FIXME: I'm not catching these errors anywhere...

    # if not global, could potentially be referenced before assignment in the try-catch block
    global result
    try:
        # Check if a or b are None
        if a is None or b is None:
            raise ValueError("<user_jb>: Invalid argument")

        # Check if a and b are floats|ints
        if not isinstance(a, (int, float)) and not isinstance(b, (int, float)):
            raise TypeError("<user_jb>: Invalid argument type")

        # Compute the sum
        result = a + b

        # Handle int overflow | underflow
        if result > float('inf') or result < float('-inf'):
            raise OverflowError("<user_jb>: Result out of range float(+|- inf)")

        # Return the sum
        # print(f"*** <user_jb>: [a = {a}] and [b = {b}], returning [result = {result}]")
        return result

    except OverflowError as overflow_error:
        print(f"[args: a={a}, b={b}, sum={result}]\n\tOverflow Error: {overflow_error}")

    except ValueError as value_error:
        print(f"[args: a={a}, b={b}, sum={result}]\n\tValue Error: {value_error}")

    except TypeError as type_error:
        print(f"[args: a={a}, b={b}, sum={result}]\n\tType Error: {type_error}")
