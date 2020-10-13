from collections import defaultdict
from abc import abstractmethod


def closure(t, x: set):
    closure_x = x.copy()
    for s, tr in x:
        if '' in t.graph[s]:
            for to in t.graph[s]['']:
                closure_x.add((to, tr))
    if closure_x == x:
        return closure_x
    return closure(t, closure_x)


class LabeledTransitionSystem:

    # transitions : list<tuple<int, str, int>>
    # tuple data meaning:
    # (t-, t_lbl, t+)
    def __init__(self, transitions: list, start, end):
        self.start = start
        self.end = end
        self.graph = defaultdict(defaultdict)
        for t_from, t_lbl, t_to in transitions:
            if t_from not in self.graph:
                self.graph[t_from] = defaultdict(list)
            if t_to not in self.graph:
                self.graph[t_to] = defaultdict(list)
            self.graph[t_from][t_lbl].append(t_to)
        pass

    # check if the LTS accepts given string
    def accept(self, s: str, v=None) -> bool:
        q_next = {(self.start, 0)}
        q_next = closure(self, q_next)
        while len(q_next):
            q = q_next
            q_next = set()
            while len(q):
                state, length = q.pop()
                if (state, length) == (self.end, len(s)):
                    return True
                if length >= len(s):
                    continue
                if s[length] in self.graph[state]:
                    for to in self.graph[state][s[length]]:
                        q_next.add((to, length + 1))
            q_next = closure(self, q_next)
        return False

    # add new state to LTS
    def add_state(self, state_id):
        if state_id in self.graph:
            raise ValueError("State already exists in LTS")
        self.graph[state_id] = defaultdict(list)

    # renumber states of LTS with the given mapping
    # @param dictionary with keys of old state numbers and values of new ones
    def renumber_states(self, renumber: dict):
        new_graph = defaultdict(defaultdict)
        for state in self.graph:
            new_graph[renumber[state]] = defaultdict(list)
            for tr in self.graph[state]:
                new_graph[renumber[state]][tr] = list(map(lambda i: renumber[i], self.graph[state][tr]))
        self.start = renumber[self.start]
        self.end = renumber[self.end]
        self.graph = new_graph


class ReX:
    @abstractmethod
    def __str__(self):
        pass

    # Method converts this ReX to LTS object
    @abstractmethod
    def assemble_lts(self) -> LabeledTransitionSystem:
        pass

    def assert_args(*args):
        for arg in args:
            if not isinstance(arg, ReX):
                raise ValueError("Argument is not a ReX")


class Eps(ReX):
    def __str__(self):
        return ""

    def assemble_lts(self) -> LabeledTransitionSystem:
        return LabeledTransitionSystem([(1, '', 2)], 1, 2)


class Just(ReX):
    def __init__(self, symbol: str):
        self.symbol = symbol

    def __str__(self):
        return self.symbol

    def assemble_lts(self) -> LabeledTransitionSystem:
        return LabeledTransitionSystem([(1, self.symbol, 2)], 1, 2)


class Repeat(ReX):
    def __init__(self, expr: ReX):
        ReX.assert_args(expr)
        self.expr = expr

    def __str__(self):
        return str(self.expr) + '*'

    def assemble_lts(self) -> LabeledTransitionSystem:
        my_lts = self.expr.assemble_lts()
        lts_size = len(my_lts.graph)
        new_start = lts_size + 1
        new_end = lts_size + 2
        my_lts.add_state(new_start)
        my_lts.add_state(new_end)
        my_lts.graph[my_lts.start][''].append(my_lts.end)
        my_lts.graph[my_lts.end][''].append(my_lts.start)
        my_lts.graph[new_start][''].append(my_lts.start)
        my_lts.graph[my_lts.end][''].append(new_end)
        my_lts.start = new_start
        my_lts.end = new_end
        return my_lts


class Alter(ReX):
    def __init__(self, expr_left: ReX, expr_right: ReX):

        ReX.assert_args(expr_left, expr_right)
        self.expr_left = expr_left
        self.expr_right = expr_right

    def __str__(self):
        return '(' + str(self.expr_left) + ',' + str(self.expr_right) + ')'

    def assemble_lts(self) -> LabeledTransitionSystem:
        my_lts_1 = self.expr_left.assemble_lts()
        my_lts_2 = self.expr_right.assemble_lts()
        lts_1_size = len(my_lts_1.graph)
        lts_2_size = len(my_lts_2.graph)
        my_lts_2.renumber_states({i: i + lts_1_size for i in my_lts_2.graph})
        new_start = lts_1_size + lts_2_size + 1
        new_end = lts_1_size + lts_2_size + 2
        my_lts_1.add_state(new_start)
        my_lts_1.add_state(new_end)
        my_lts_1.graph = {**my_lts_1.graph, **my_lts_2.graph}
        my_lts_1.graph[new_start][''].append(my_lts_1.start)
        my_lts_1.graph[new_start][''].append(my_lts_2.start)
        my_lts_1.graph[my_lts_1.end][''].append(new_end)
        my_lts_1.graph[my_lts_2.end][''].append(new_end)
        my_lts_1.start = new_start
        my_lts_1.end = new_end
        return my_lts_1


class Concat(ReX):
    def __init__(self, expr_left: ReX, expr_right: ReX):
        ReX.assert_args(expr_left, expr_right)

        self.expr_left = expr_left
        self.expr_right = expr_right

    def __str__(self):
        return '(' + str(self.expr_left) + '|' + str(self.expr_right) + ')'

    def assemble_lts(self) -> LabeledTransitionSystem:
        my_lts_1 = self.expr_left.assemble_lts()
        my_lts_2 = self.expr_right.assemble_lts()
        lts_1_size = len(my_lts_1.graph)
        my_lts_2.renumber_states({i: i + lts_1_size for i in my_lts_2.graph})
        my_lts_1.graph = {**my_lts_1.graph, **my_lts_2.graph}
        my_lts_1.graph[my_lts_1.end][''].append(my_lts_2.start)
        my_lts_1.end = my_lts_2.end
        return my_lts_1


# Function to solve problem
#
# @param ReX object
# @return created LTS
def ReX2LTS(regex: ReX):
    return regex.assemble_lts()


# ReX accepts arbitrary combinations of "aa" and "b"
re = Repeat(Alter(Concat(Just('a'), Just('a')), Just('b')))
print("1. ReX accepts arbitrary combinations of \"aa\" and \"b\":")
print(re)
re_lts = ReX2LTS(re)
for s in ["aab", "aabaa", "aaa", "aac", ""]:
    print("Accepts " + s + " ? : " + ("YES" if re_lts.accept(s) else "NO"))

# ReX accepts non-decreasing successions of 1, 2 and 3
re = Concat(Concat(Repeat(Just('1')), Repeat(Just('2'))), Repeat(Just('3')))
print("2. ReX accepts non-decreasing successions of 1, 2 and 3:")
print(re)
re_lts = ReX2LTS(re)
for s in ["123", "1122233", "321", "1111"]:
    print("Accepts " + s + " ? : " + ("YES" if re_lts.accept(s) else "NO"))
