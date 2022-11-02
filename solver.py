import sys
import puzz
import pdqpq

GOAL_STATE = puzz.EightPuzzleBoard("012345678")

def solve_puzzle(start_state, flavor):
    """Perform a search to find a solution to a puzzle.
    
    Args:
        start_state (EightPuzzleBoard): the start state for the search
        flavor (str): tag that indicate which type of search to run.  Can be one of the following:
            'bfs' - breadth-first search
            'ucost' - uniform-cost search
            'greedy-h1' - Greedy best-first search using a misplaced tile count heuristic
            'greedy-h2' - Greedy best-first search using a Manhattan distance heuristic
            'greedy-h3' - Greedy best-first search using a weighted Manhattan distance heuristic
            'astar-h1' - A* search using a misplaced tile count heuristic
            'astar-h2' - A* search using a Manhattan distance heuristic
            'astar-h3' - A* search using a weighted Manhattan distance heuristic
    
    Returns: 
        A dictionary containing describing the search performed, containing the following entries:
        'path' - list of 2-tuples representing the path from the start to the goal state (both 
            included).  Each entry is a (str, EightPuzzleBoard) pair indicating the move and 
            resulting successor state for each action.  Omitted if the search fails.
        'path_cost' - the total cost of the path, taking into account the costs associated with 
            each state transition.  Omitted if the search fails.
        'frontier_count' - the number of unique states added to the search frontier at any point 
            during the search.
        'expanded_count' - the number of unique states removed from the frontier and expanded 
            (successors generated)

    """
    if flavor.find('-') > -1:
        strat, heur = flavor.split('-')
    else:
        strat, heur = flavor, None

    if strat == 'bfs':
        return BreadthFirstSolver().solve(start_state)
    elif strat == 'ucost':
        return UniformCostSolver().solve(start_state)
    elif strat == 'greedy':
        if heur == 'h1':
            return GreedySolver(heuristic_1).solve(start_state)
        elif heur == 'h2':
            return GreedySolver(heuristic_2).solve(start_state)
        elif heur == 'h3':
            return GreedySolver(heuristic_3).solve(start_state)  
    elif strat == 'astar':
        if heur == 'h1':
            return AStarSolver(heuristic_1).solve(start_state)
        elif heur == 'h2':
            return AStarSolver(heuristic_2).solve(start_state)
        elif heur == 'h3':
            return AStarSolver(heuristic_3).solve(start_state)  
    else:
        raise ValueError("Unknown search flavor '{}'".format(flavor))


class BreadthFirstSolver:
    """Implementation of Breadth-First Search based puzzle solver"""

    def __init__(self):
        self.goal = GOAL_STATE
        self.parents = {}  # state -> parent_state
        self.frontier = pdqpq.FifoQueue()
        self.explored = set()
        self.frontier_count = 0  # increment when we add something to frontier
        self.expanded_count = 0  # increment when we pull something off frontier and expand
    
    def solve(self, start_state):
        """Carry out the search for a solution path to the goal state.
        
        Args:
            start_state (EightPuzzleBoard): start state for the search 
        
        Returns:
            A dictionary describing the search from the start state to the goal state.

        """
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            succs = self.expand_node(node)
            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node

                    # BFS checks for goal state _before_ adding to frontier
                    if succ == self.goal:
                        return self.get_results_dict(succ)
                    else:
                        self.add_to_frontier(succ)

        # if we get here, the search failed
        return self.get_results_dict(None) 

    def add_to_frontier(self, node):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def get_results_dict(self, state):
        """Construct the output dictionary for solve_puzzle()
        
        Args:
            state (EightPuzzleBoard): final state in the search tree
        
        Returns:
            A dictionary describing the search performed (see solve_puzzle())

        """
        results = {}
        results['frontier_count'] = self.frontier_count
        results['expanded_count'] = self.expanded_count
        if state:
            results['path_cost'] = self.get_cost(state)
            path = self.get_path(state)
            moves = ['start'] + [ path[i-1].get_move(path[i]) for i in range(1, len(path)) ]
            results['path'] = list(zip(moves, path))
        return results

    def get_path(self, state):
        """Return the solution path from the start state of the search to a target.
        
        Results are obtained by retracing the path backwards through the parent tree to the start
        state for the serach at the root.
        
        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            A list of EightPuzzleBoard objects representing the path from the start state to the
            target state

        """
        path = []
        while state is not None:
            path.append(state)
            state = self.parents[state]
        path.reverse()
        return path

    def get_cost(self, state): 
        """Calculate the path cost from start state to a target state.
        
        Transition costs between states are equal to the square of the number on the tile that 
        was moved. 

        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            Integer indicating the cost of the solution path

        """
        cost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            tile = path[i].get_tile(x, y)        
            cost += int(tile)**2
        return cost

class AStarSolver(BreadthFirstSolver):
    def __init__(self, heuristic):
        super().__init__()
        self.frontier = pdqpq.PriorityQueue()
        self.heuristic = heuristic

    def solve(self, start_state):
        """Carry out the search for a solution path to the goal state.
        
        Args:
            start_state (EightPuzzleBoard): start state for the search 
        
        Returns:
            A dictionary describing the search from the start state to the goal state.

        """
        self.parents[start_state] = None
        self.add_to_frontier(start_state, self.heuristic(start_state))

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if (node == self.goal):
                return self.get_results_dict(node)
            succs = self.expand_node(node)
            prev_cost = self.get_cost(node)
            for move, succ in succs.items():
                index = str(node).index('0')
                num = int(str(succ)[index])
                cost = self.heuristic(succ) + prev_cost + num ** 2
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, cost)
                elif (succ in self.frontier and self.frontier.get(succ) > cost):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, cost)
                    self.frontier_count -= 1
        # if we get here, the search failed
        return self.get_results_dict(None)    
            
    def add_to_frontier(self, node, cost):
        """Add state and cost and heuristic to frontier and increase the frontier count."""
        self.frontier.add(node, cost)
        self.frontier_count += 1
        
    # def get_cost(self, state):
    #     """Calculate the path cost from start state to a target state.
        
    #     Transition costs between states are equal to the square of the number on the tile that 
    #     was moved. 

    #     Args:
    #         state (EightPuzzleBoard): target state in the search tree
        
    #     Returns:
    #         Integer indicating the cost of the solution path

    #     """
    #     cost = self.heuristic(state)
    #     path = self.get_path(state)
    #     for i in range(1, len(path)):
    #         x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
    #         tile = path[i].get_tile(x, y)        
    #         cost += int(tile)**2 + self.heuristic(path[i])
    #     return cost

    
class GreedySolver(BreadthFirstSolver):
    def __init__(self, heuristic):
        super().__init__()
        self.frontier = pdqpq.PriorityQueue()
        self.heuristic = heuristic

    def solve(self, start_state):
        """Carry out the search for a solution path to the goal state.
        
        Args:
            start_state (EightPuzzleBoard): start state for the search 
        
        Returns:
            A dictionary describing the search from the start state to the goal state.

        """
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if (node == self.goal):
                return self.get_results_dict(node)
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node

                    # BFS checks for goal state _before_ adding to frontier
                    self.add_to_frontier(succ)
        # if we get here, the search failed
        return self.get_results_dict(None)    
            
    def add_to_frontier(self, node):
        """Add state and cost and heuristic to frontier and increase the frontier count."""
        self.frontier.add(node, self.heuristic(node))
        self.frontier_count += 1
    def get_cost(self, state):
        """Calculate the path cost from start state to a target state.
        
        Transition costs between states are equal to the square of the number on the tile that 
        was moved. 

        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            Integer indicating the cost of the solution path

        """
        cost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            tile = path[i].get_tile(x, y)        
            cost += int(tile)**2 
        return cost


#################################################################################
class UniformCostSolver(BreadthFirstSolver):
    def __init__(self):
        super().__init__()
        self.frontier = pdqpq.PriorityQueue()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.frontier.add(start_state)
        self.frontier_count += 1

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if node == self.goal:
                return self.get_results_dict(node)
                        
            prev_cost = self.get_cost(node)

            succs = self.expand_node(node)
            for move, succ in succs.items():
                index = str(node).index('0')
                num = int(str(succ)[index])
                cost = prev_cost + num**2
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.frontier.add(succ, cost)
                    self.frontier_count += 1
                elif (succ in self.frontier) and (self.frontier.get(succ) > cost):
                    self.parents[succ] = node
                    self.frontier.add(succ, cost)

        # if we get here, the search failed
        return self.get_results_dict(None) 
    
def print_table(flav__results, include_path=False):
    """Print out a comparison of search strategy results.

    Args:
        flav__results (dictionary): a dictionary mapping search flavor tags result statistics. See
            solve_puzzle() for detail.
        include_path (bool): indicates whether to include the actual solution paths in the table

    """
    result_tups = sorted(flav__results.items())
    c = len(result_tups)
    na = "{:>12}".format("n/a")
    rows = [  # abandon all hope ye who try to modify the table formatting code...
        "flavor  " + "".join([ "{:>12}".format(tag) for tag, _ in result_tups]),
        "--------" + ("  " + "-"*10)*c,
        "length  " + "".join([ "{:>12}".format(len(res['path'])) if 'path' in res else na 
                                for _, res in result_tups ]),
        "cost    " + "".join([ "{:>12,}".format(res['path_cost']) if 'path_cost' in res else na 
                                for _, res in result_tups ]),
        "frontier" + ("{:>12,}" * c).format(*[res['frontier_count'] for _, res in result_tups]),
        "expanded" + ("{:>12,}" * c).format(*[res['expanded_count'] for _, res in result_tups])
    ]
    if include_path:
        rows.append("path")
        longest_path = max([ len(res['path']) for _, res in result_tups if 'path' in res ] + [0])
        print("longest", longest_path)
        for i in range(longest_path):
            row = "        "
            for _, res in result_tups:
                if len(res.get('path', [])) > i:
                    move, state = res['path'][i]
                    row += " " + move[0] + " " + str(state)
                else:
                    row += " "*12
            rows.append(row)
    print("\n" + "\n".join(rows), "\n")

def heuristic_1(start_state):
    # Number of misplaced tiles
    
    misplaced_tiles = 0
    for i in range(0, 9):
        if start_state._board[i] != GOAL_STATE._board[i]:
            misplaced_tiles += 1
    return misplaced_tiles
        
def heuristic_2(start_state):
    # Manhattan Distance
    manhattan_distance = 0
    for i in range(0, 9):
        vertical = abs(start_state._board.index(str(i)) // 3 - GOAL_STATE._board.index(str(i)) // 3)
        horizontal = abs(start_state._board.index(str(i)) % 3 - GOAL_STATE._board.index(str(i)) % 3)
        manhattan_distance += vertical + horizontal
    return manhattan_distance
        
def heuristic_3(start_state):
    manhattan_distance = 0
    for i in range(0, 9):
        vertical = abs(start_state._board.index(str(i)) // 3 - GOAL_STATE._board.index(str(i)) // 3)
        horizontal = abs(start_state._board.index(str(i)) % 3 - GOAL_STATE._board.index(str(i)) % 3)
        manhattan_distance += (vertical + horizontal) * i * i
    return manhattan_distance
        
    
def get_test_puzzles():
    """Return sample start states for testing the search strategies.
    
    Returns:
        A tuple containing three EightPuzzleBoard objects representing start states that have an
        optimal solution path length of 3-5, 10-15, and >=25 respectively.
    
    """ 
    # Note: test cases can be hardcoded, and are not required to be programmatically generated.
    #
    # fill in function body here
    #    
    
    return (puzz.EightPuzzleBoard("312475608"), puzz.EightPuzzleBoard("325718046"), puzz.EightPuzzleBoard("867254301"))


############################################

if __name__ == '__main__':

    # parse the command line args
    start = puzz.EightPuzzleBoard(sys.argv[1])
    if sys.argv[2] == 'all':
        flavors = ['bfs', 'ucost', 'greedy-h1', 'greedy-h2', 
                   'greedy-h3', 'astar-h1', 'astar-h2', 'astar-h3']
    else:
        flavors = sys.argv[2:]

    # run the search(es)
    results = {}
    for flav in flavors:
        print("solving puzzle {} with {}".format(start, flav))
        results[flav] = solve_puzzle(start, flav)

    print_table(results, include_path=True)  # change to True to see the paths!
