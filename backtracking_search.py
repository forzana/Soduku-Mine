# Forzana Rime (fkr206) & Mohammed Uddin (msu227)
# CS-GY 6613 AI Project 2

import os
from typing import Any, List, Dict, Tuple
from heapq import *
from copy import deepcopy

input_directory_path = "./input" # The location of the input files
output_directory_path = "./output" # Where the output files will go

class ConstraintSatisfactionProblem:
    
    def __init__(self, matrix: List[List[int]]):
        self.matrix = matrix
        self.constraints = {
            "row": {},
            "column": {},
            "block": {},
            "eight": {}
        }
        self.neighbors = self.get_neighbors(matrix) # Find all the variables that need to be assigned and their neighbors
        self.variables = self.neighbors.keys() # The variables that need to be assigned
        self.remaining_domain_stack = [{key:{0,1} for key in self.neighbors}] # Domain vals for each variable

    
    def get_neighbors(self, matrix: List[List[int]]) -> Dict:
        
        neighbors = {}
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    neighbors[(i,j)] = []
        
        for key in neighbors:
            row_constraints = self.get_row_constraints(matrix, i, j)
            self.constraints["row"][key] = row_constraints
            neighbors[key].extend(row_constraints)

            col_constraints = self.get_col_constraints(matrix, i, j)
            self.constraints["column"][key] = col_constraints
            neighbors[key].extend(col_constraints)

            block_constraints = self.get_block_constraints(matrix, i, j)
            self.constraints["block"][key] = block_constraints
            neighbors[key].extend(block_constraints)

            eight_neighbor_constraints = self.get_eight_neighbor_constraints(matrix, i, j)
            self.constraints["eight"][key] = eight_neighbor_constraints
            neighbors[key].extend(eight_neighbor_constraints)
            
            neighbors[key] = list(set(neighbors[key]))

        return neighbors


    def get_row_constraints(self, matrix: List[List[int]], i: int, j: int) -> List[Tuple[int]]:
        return [ (i,k) for k in range(len(matrix)) if matrix[i][k] == 0 and j != k]

    def get_col_constraints(self, matrix: List[List[int]], i: int, j: int) -> List[Tuple[int]]:
        return [ (k,j) for k in range(len(matrix)) if matrix[k][j] == 0 and k != i]
    
    def get_block_constraints(self, matrix: List[List[int]], i: int, j: int) -> List[Tuple[int]]:

        start_row = (i // 3) * 3
        start_col = (j // 3) * 3

        constraints = []
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if matrix[r][c] == 0 and r != i and c != j:
                    constraints.append((r,c))

        return constraints

        
    def get_eight_neighbor_constraints(self, matrix: List[List[int]], i: int, j: int) -> List[Tuple[int]]:

        constraints = []

        # All relative moves: 8 directions (king moves in chess)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # up-left, up, up-right
            (0, -1),           (0, 1),   # left,       right
            (1, -1),  (1, 0),  (1, 1)    # down-left, down, down-right
        ]

        for di, dj in directions:
            ni, nj = i + di, j + dj
            # Check bounds
            if 0 <= ni < 9 and 0 <= nj < 9 and matrix[ni][nj] > 0:
                for ki, kj in directions:
                    x, y = ni + ki, nj + kj
                    if 0 <= x < 9 and 0 <= y < 9 and matrix[x][y] == 0 and x != i and y != j:
                        constraints.append((x,y))
        return constraints
        
    # If all variables are assigned, then the assignment is complete
    def is_complete(self, assignment):
        if len(self.variables) == len(assignment):
            return True
        return False

    # Use the MRV heuristic and Degree heuristic to select the next variable
    def select_unassigned_var(self, assignment):
        all_unassigned = [var for var in self.variables if var not in assignment]

        # Minimum remaining values
        min_val = min(len(self.remaining_domain_stack[var]) for var in all_unassigned)
        mrv_unassigned = [var for var in all_unassigned if len(self.remaining_domain_stack[var]) == min_val]
        if len(mrv_unassigned) == 1:
            return mrv_unassigned[0]
        
        # Degree heuristic
        max_neighbors = 0
        var_with_max = None
        for var in mrv_unassigned:
            unassigned_var_neighbors = sum(1 for neighbor in self.neighbors[var] if neighbor in all_unassigned)
            if unassigned_var_neighbors >= max_neighbors:
                unassigned_var_neighbors = unassigned_var_neighbors
                var_with_max = var
        return var_with_max

    # Use the order {0, 1} to find remaining domain values
    def order_domain_values(self, var):
        return [val for val in {0, 1} if val in self.remaining_domain_stack[var]]

    def satisfies_row_constraint(self, var, assignment):
        unassigned_count = 0
        assigned_count = 0

        for r, c in self.constraints["row"][var]:
            if (r, c) in assignment:
                assigned_count += 1
            else:
                unassigned_count += 1
        
        if assigned_count > 3 or assigned_count + unassigned_count < 3:
            return False
        
        return True

    def satisfies_column_constraint(self, var, assignment):
        unassigned_count = 0
        assigned_count = 0

        for r, c in self.constraints["column"][var]:
            if (r, c) in assignment:
                assigned_count += 1
            else:
                unassigned_count += 1
        
        if assigned_count > 3 or assigned_count + unassigned_count < 3:
            return False
        
        return True

    def satisfies_block_constraint(self, var, assignment):
        unassigned_count = 0
        assigned_count = 0

        for r, c in self.constraints["block"][var]:
            if (r, c) in assignment:
                assigned_count += 1
            else:
                unassigned_count += 1
        
        if assigned_count > 3 or assigned_count + unassigned_count < 3:
            return False
        
        return True

    def satisfies_eight_neighbor_constraints(self, var, assignment):


    def satisfies_constraints(self, var, assignment):
        return self.satisfies_row_constraint(var, assignment) and self.satisfies_column_constraint(var, assignment) and self.satisfies_block_constraint(var, assignment) and self.satisfies_eight_neighbor_constraints(var, assignment)

    # Check if the partial assignment is consistent with all contraints
    def is_consistent(self, var, value, assignment):
        assignment[var] = value
        if self.satisfies_constraints(var, assignment):
            del assignment[var]
            return True
        else:
            del assignment[var]
            return False

    
    def forward_check(self, var, assignment):


    def backtrack(self, assignment):
        if self.is_complete(assignment):
            return assignment
        var = self.select_unassigned_var(assignment)
        for value in self.order_domain_values(var):
            # TO-DO
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                if self.forward_check(var, assignment):
                    result = self.backtrack(assignment)
                    if result:
                        return result
                del assignment[var]

    def backtracking_search(self):
        assignment = {}
        return self.backtrack(assignment)

def parse_input():
    # Process each input text file
    for entry_name in os.listdir(input_directory_path):
        full_path = os.path.join(input_directory_path, entry_name)
        if os.path.isfile(full_path):
            # Open the file and grab the contents
            test_file = open(full_path)
            test_file_content = test_file.read().strip()
            test_file.close()

            # Parse the initial state
            initial_state = list(map(lambda x: x.split(' '), test_file_content.split('\n')))

            # Yield the initial state matrix and the problem number
            yield initial_state, entry_name.replace('Input', '').replace('.txt', '')

def main():
    # For each problem (input file)
    for initial_state, problem_number in parse_input():
        output_file = open(f"{output_directory_path}/output{problem_number}.txt", "w") # Write to an output file with the naming convention required
        csp = ConstraintSatisfactionProblem(initial_state) # Perform the backtracking search
        goal_node, node_count = csp.backtracking_search()
        if not goal_node: # Search was not successful
            print("failure")
            continue
        output_file.write(str(goal_node.path_cost) + "\n") # path cost of the goal node will tell us the depth
        output_file.write(str(node_count) + "\n") # Number of nodes generated
        output_file.close()

main()