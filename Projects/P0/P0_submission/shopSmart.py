# shopSmart.py
# ------------
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
Here's the intended output of this script, once you fill it in:

Welcome to shop1 fruit shop
Welcome to shop2 fruit shop
For orders:  [('apples', 1.0), ('oranges', 3.0)] best shop is shop1
For orders:  [('apples', 3.0)] best shop is shop2
"""
from __future__ import print_function
import shop


def shopSmart(orderList, fruitShops):
    """
        orderList: List of (fruit, numPound) tuples
        fruitShops: List of FruitShops
    """
    # Order list is empty
    if not orderList:
        raise ValueError("<user_jb>: The order list is empty")

    # Fruit shops list is empty
    if not fruitShops:
        raise ValueError("<user_jb>: The fruit shops list is empty")

    # Validate orderList structure
    for item in orderList:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"<user_jb>: Invalid order format: {item}. Each order must be a (fruit, numPounds) tuple.")
        fruit, numPounds = item
        if not isinstance(fruit, str) or not isinstance(numPounds, (int, float)) or numPounds < 0:
            raise TypeError(f"<user_jb>: Invalid order entry: {item}. Fruit must be a string and numPounds must be a non-negative number.")

    # Set up variables
    minCost = float('inf')
    bestShop = None

    for shop in fruitShops:
        # Probably unnecessary, but we want to check if the method exists
        if not hasattr(shop, 'getPriceOfOrder') or not callable(getattr(shop, 'getPriceOfOrder')):
            raise NotImplementedError("<user_jb>: The implementation for the \"getPriceOfOrder()\" function is missing.")

        # Doesn't raise any errors in <shop.py>
        totalCost = shop.getPriceOfOrder(orderList)


        if totalCost < minCost:
            minCost = totalCost
            bestShop = shop

    return bestShop


if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orders = [('apples', 1.0), ('oranges', 3.0)]
    dir1 = {'apples': 2.0, 'oranges': 1.0}
    shop1 = shop.FruitShop('shop1', dir1)
    dir2 = {'apples': 1.0, 'oranges': 5.0}
    shop2 = shop.FruitShop('shop2', dir2)
    shops = [shop1, shop2]

    #  1-st try-catch block
    try:

        print("For orders ", orders, ", the best shop is", shopSmart(orders, shops).getName())

    # Catch separately to avoid confusion
    except ValueError as value_error:
        print(f"[args: orders={orders}, shops={shops}]\n\tValue Error: {value_error}")

    # Catch separately to avoid confusion
    except TypeError as type_error:
        print(f"[args: orders={orders}, shops={shops}]\n\tType Error: {type_error}")

    # Catch separately to avoid confusion
    except NotImplementedError as not_implemented_error:
        print(f"[args: orders={orders}, shops={shops}]\n\tNot implemented Error: {not_implemented_error}")

    # 2-nd try-catch block
    try:
        orders = [('apples', 3.0)]
        print("For orders: ", orders, ", the best shop is", shopSmart(orders, shops).getName())
    except ValueError as value_error:
        print(f"[args: orders={orders}, shops={shops}]\n\tValue Error: {value_error}")

    # Catch separately to avoid confusion
    except TypeError as type_error:
        print(f"[args: orders={orders}, shops={shops}]\n\tType Error: {type_error}")

    # Catch separately to avoid confusion
    except NotImplementedError as not_implemented_error:
        print(f"[args: orders={orders}, shops={shops}]\n\tNot implemented Error: {not_implemented_error}")
