# buyLotsOfFruit.py
# -----------------
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
To run this script, type

  python buyLotsOfFruit.py

Once you have correctly implemented the buyLotsOfFruit function,
the script should produce the output:

Cost of [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)] is 12.25
"""
from __future__ import print_function

fruitPrices = {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75,
               'limes': 0.75, 'strawberries': 1.00}


def buyLotsOfFruit(orderList):
    """
        orderList: List of (fruit, numPounds) tuples

    Returns cost of order
    """

    if not orderList:
        raise ValueError("<user_jb>: Order list is empty.")

    totalCost = 0.0

    for item, quantity in orderList:
        # Check if item even in fruitPrices
        if item not in fruitPrices:
            raise ValueError("<user_jb>: Item not available in fruit list.")

        # Check if quantity is a float|int
        if not isinstance(quantity, (int, float)):
            raise TypeError("<user_jb>: Invalid quantity type.")

        # Check if quantity is positive
        if quantity < 0:
            raise ValueError("<user_jb>: Quantity cannot be negative.")

        # Compute total cost iteratively
        totalCost += fruitPrices[item] * quantity

    # Return totalCost
    return totalCost


# Main Method
# Modified: <user_jb> added a try-catch block to gracefully handle errors
if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orderList = [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)]
    try:
        totalCost = buyLotsOfFruit(orderList)
        print('Cost of', orderList, 'is', buyLotsOfFruit(orderList))

    # Catch separately to avoid confusion
    except TypeError as type_error:
        print(f"[args: orderList={orderList}]\n\tType Error: {type_error}")

    # Catch separately to avoid confusion
    except ValueError as value_error:
        print(f"[args: orderList={orderList}]\n\tValue Error: {value_error}")
