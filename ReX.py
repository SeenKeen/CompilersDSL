
# TASK 1

from abc import abstractmethod


class ReX:
    @abstractmethod
    def __str__(self):
        pass

    def assert_args(*args):
        for arg in args:
            if not isinstance(arg, ReX):
                raise ValueError("Argument is not a ReX")


class Eps(ReX):
    def __str__(self):
        return ""


class Just(ReX):
    def __init__(self, symbol: str):
        self.symbol = symbol

    def __str__(self):
        return self.symbol


class Repeat(ReX):
    def __init__(self, expr: ReX):
        ReX.assert_args(expr)
        self.expr = expr

    def __str__(self):
        return str(self.expr) + '*'


class Alter(ReX):
    def __init__(self, expr_left: ReX, expr_right: ReX):

        ReX.assert_args(expr_left, expr_right)
        self.expr_left = expr_left
        self.expr_right = expr_right

    def __str__(self):
        return '(' + str(self.expr_left) + ',' + str(self.expr_right) + ')'


class Concat(ReX):
    def __init__(self, expr_left: ReX, expr_right: ReX):
        ReX.assert_args(expr_left, expr_right)

        self.expr_left = expr_left
        self.expr_right = expr_right

    def __str__(self):
        return '(' + str(self.expr_left) + '|' + str(self.expr_right) + ')'


#  (this is realization by strongly typed expressions)
#  found this more reliable,
#  BUT
#  is does not use smart constructor.

#  There 5 types of ReX:

#  1. Eps() - epsilon
#  2. Just(c) - just symbol
#  3. Repeat(ReX) - repetition
#  4. Alter(ReX, ReX) - alternate
#  5. Concat(ReX, ReX) - concatenate


# here is an example :

rex = Alter(Just('a'), Concat(Repeat(Just('b')), Eps()))
print(rex)
