from collections import defaultdict


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
            self.graph[t_from][t_lbl].append(t_to)
        pass

    def accept(self, s: str, v=None) -> bool:
        if v is None:
            #  if no state vertex given
            v = self.start
        if len(s) == 0:
            if self.end == v:
                return True
            else:
                return False
        if s[0] not in self.graph[v]:
            return False
        for to in self.graph[v][s[0]]:
            if self.accept(s[1:], to):
                return True
        return False


trans = [(1, '0', 2)]
lts = LabeledTransitionSystem(trans, 1, 2)
print(lts.accept("00"))
