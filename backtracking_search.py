# Forzana Rime (fkr206) & Mohammed Uddin (msu227)
# CS-GY 6613 AI Project 2

import os
from typing import List, Dict, Tuple
from heapq import *

input_directory_path = "./input" # The location of the input files
output_directory_path = "./output" # Where the output files will go

class ConstraintSatisfactionProblem:
    
    def __init__(self, matrix: List[List[int]]):
        self.matrix = matrix
        self.constraints = { # neighbors indexed by constraint type and variable
            "row": {}, 
            "column": {},
            "block": {},
            "eight": {}
        }
        self.neighbors = self.get_neighbors(matrix) # Find all the variables that need to be assigned and their neighbors
        self.variables = self.neighbors.keys() # The variables that need to be assigned
        self.remaining_domain_stack = [{key:[0,1] for key in self.neighbors}] # Domain vals for each variable at current level
        self.nodes = 0

    
    def get_neighbors(self, matrix: List[List[int]]) -> Dict:
        
        neighbors = {}
        # We're getting a list of all variables
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    neighbors[(i,j)] = []
        
        # We iterate thru each variable
        for key in neighbors:
            i, j = key

            # We get neighbors that are by row constraint
            row_constraints = self.get_row_constraints(matrix, i, j)
            self.constraints["row"][key] = row_constraints
            neighbors[key].extend(row_constraints)

            # We get neighbors that are by the column constraint
            col_constraints = self.get_col_constraints(matrix, i, j)
            self.constraints["column"][key] = col_constraints
            neighbors[key].extend(col_constraints)

            # Get neighbors that are by the block constraint
            block_constraints = self.get_block_constraints(matrix, i, j)
            self.constraints["block"][key] = block_constraints
            neighbors[key].extend(block_constraints)

            # Get neighbors that are by the eight neighbor constraint
            eight_neighbor_constraints = self.get_eight_neighbor_constraints(matrix, i, j)
            self.constraints["eight"][key] = eight_neighbor_constraints
            neighbors[key].extend(eight_neighbor_constraints)
            
            # all the above calls might have added dupe neighbors, so return deduped list
            neighbors[key] = list(set(neighbors[key]))

        return neighbors


    def get_row_constraints(self, matrix: List[List[int]], i: int, j: int) -> List[Tuple[int]]:
        return [ (i,k) for k in range(len(matrix)) if matrix[i][k] == 0 and j != k]

    def get_col_constraints(self, matrix: List[List[int]], i: int, j: int) -> List[Tuple[int]]:
        return [ (k,j) for k in range(len(matrix)) if matrix[k][j] == 0 and k != i]
    
    def get_block_constraints(self, matrix: List[List[int]], i: int, j: int) -> List[Tuple[int]]:
        
        # Get top left cell in block
        start_row = (i // 3) * 3
        start_col = (j // 3) * 3

        constraints = []
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                # Add every variable in block except current variable we are processing
                if matrix[r][c] == 0 and (i, j) != (r, c):
                    constraints.append((r,c))

        return constraints

        
    def get_eight_neighbor_constraints(self, matrix: List[List[int]], i: int, j: int) -> List[Tuple[int]]:

        constraints = set()

        # All relative moves: 8 directions (king moves in chess)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # up-left, up, up-right
            (0, -1),           (0, 1),   # left,       right
            (1, -1),  (1, 0),  (1, 1)    # down-left, down, down-right
        ]

        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            # Check bounds
            # Check if its constraining value
            if 0 <= ni < 9 and 0 <= nj < 9 and matrix[ni][nj] > 0:
                # Look at variables surrounding constraining value
                for ki, kj in directions:
                    x, y = ni + ki, nj + kj
                    # Check bounds
                    # Check if its a variable
                    # Check if its not current variable we are processing
                    if 0 <= x < 9 and 0 <= y < 9 and matrix[x][y] == 0 and (x, y) != (i, j):
                        constraints.add((x,y))
        # We want a unique list of variables cause there might be dupes
        return list(constraints)
        
    # If all variables are assigned, then the assignment is complete
    def is_complete(self, assignment):
        if len(self.variables) == len(assignment):
            return True
        return False

    # Use the MRV heuristic and Degree heuristic to select the next variable
    def select_unassigned_var(self, assignment):
        all_unassigned = [var for var in self.variables if var not in assignment]

        # Minimum remaining values
        # remaining_domain_stack contains current possible domain for all variables
        # indexing by var gives current possible domain of the unassigned variable
        # we take the size of each and take minimum
        min_val = min(len(self.remaining_domain_stack[-1][var]) for var in all_unassigned)

        # Filter by minimum domain size
        mrv_unassigned = [var for var in all_unassigned if len(self.remaining_domain_stack[-1][var]) == min_val]
        # Return var if there is only 1 left after MRV
        if len(mrv_unassigned) == 1:
            return mrv_unassigned[0]
        
        # Degree heuristic
        max_neighbors = 0
        var_with_max = None
        for var in mrv_unassigned:
            # count number of unassigned neighbors the current variable has
            unassigned_var_neighbors = sum(1 for neighbor in self.neighbors[var] if neighbor in all_unassigned)
            if unassigned_var_neighbors >= max_neighbors:
                max_neighbors = unassigned_var_neighbors
                var_with_max = var
        return var_with_max

    # Use the order {0, 1} to find remaining domain values
    def order_domain_values(self, var):
        return [val for val in [0, 1] if val in self.remaining_domain_stack[-1][var]]

    # Check the row constraint
    def satisfies_row_constraint(self, var, assignment):
        unassigned_count = 0
        assigned_count = 0

        # Count the current var
        assigned_count += assignment[var]
        
        # Go through the neighbors by row constraint
        for r, c in self.constraints["row"][var]:
            if (r, c) in assignment:
                assigned_count += assignment[(r, c)]
            else:
                unassigned_count += 1
        
        # If there are more than 3 bombs or not possible to have 3 bombs, then return False
        if assigned_count > 3 or assigned_count + unassigned_count < 3:
            return False
        
        return True

    # Check the column constraint
    def satisfies_column_constraint(self, var, assignment):
        unassigned_count = 0
        assigned_count = 0

        # Count the current var
        assigned_count += assignment[var]

        # Go through the neighbors by column constraint
        for r, c in self.constraints["column"][var]:
            if (r, c) in assignment:
                assigned_count += assignment[(r, c)]
            else:
                unassigned_count += 1
        
        # If there are more than 3 bombs or not possible to have 3 bombs, then return False
        if assigned_count > 3 or assigned_count + unassigned_count < 3:
            return False
        
        return True

    # Check the block constraint
    def satisfies_block_constraint(self, var, assignment):
        unassigned_count = 0
        assigned_count = 0

        # Count the current var
        assigned_count += assignment[var]
    
        # Go through the neighbors by block constraint
        for r, c in self.constraints["block"][var]:
            if (r, c) in assignment:
                assigned_count += assignment[(r, c)]
            else:
                unassigned_count += 1
        
        # If there are more than 3 bombs or not possible to have 3 bombs, then return False
        if assigned_count > 3 or assigned_count + unassigned_count < 3:
            return False
        
        return True

    # Check the eight neighbors constraint
    def satisfies_eight_neighbor_constraints(self, var, assignment):
        (i, j) = var

        # All relative moves: 8 directions
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # up-left, up, up-right
            (0, -1),           (0, 1),   # left,       right
            (1, -1),  (1, 0),  (1, 1)    # down-left, down, down-right
        ]
        
        # Look for any constraints in the 8 cells surrounding this var
        for di, dj in directions:
            ni, nj = i + di, j + dj
            assigned_count = 0
            unassigned_count = 0
            # Check bounds
            if 0 <= ni < 9 and 0 <= nj < 9 and self.matrix[ni][nj] > 0:
                # The constraining value is the threshold
                threshold = self.matrix[ni][nj]

                # Look at surrounding 8 cells 
                for ki, kj in directions:
                    x, y = ni + ki, nj + kj
                    if 0 <= x < 9 and 0 <= y < 9 and self.matrix[x][y] == 0: 
                        if (x, y) in assignment:
                            assigned_count += assignment[(x, y)]
                        else:
                            unassigned_count += 1

                # Verify against the threshold
                if assigned_count > threshold or assigned_count + unassigned_count < threshold:
                    return False
                
        return True

    # Returns whether all constraints are satisfied
    def satisfies_constraints(self, var, assignment):
        return self.satisfies_row_constraint(var, assignment) and self.satisfies_column_constraint(var, assignment) and self.satisfies_block_constraint(var, assignment) and self.satisfies_eight_neighbor_constraints(var, assignment)

    # Check if the partial assignment is consistent with all contraints
    def is_consistent(self, var, value, assignment):
        # We temporarily assign it to make constraint checking easier, then delete the assignment before returning
        assignment[var] = value
        if self.satisfies_constraints(var, assignment):
            del assignment[var]
            return True
        else:
            del assignment[var]
            return False

    # Inference function is forward checking
    def forward_check(self, var, assignment):
        # Make a copy of the current possible domain vals for the variables
        current_domain_vals = {key:list(self.remaining_domain_stack[-1][key]) for key in self.remaining_domain_stack[-1]}

        # Filter by vars in the neighbors that have not yet been assigned
        var_neighbors = {key: current_domain_vals[key] for key in self.neighbors[var] if key not in assignment}
        
        # Go through each of the neighbors for the current var
        for neighbor in var_neighbors.keys():
            # Grab the domain of the neighbor
            domain = list(var_neighbors[neighbor])

            # Temp assign each domain and check to see if its consistent
            for d in domain:
                # If the assignment is not consistent, reduce the domain by deleting d
                if not self.is_consistent(neighbor, d, assignment):
                    current_domain_vals[neighbor] = [val for val in current_domain_vals[neighbor] if val != d]
                # If domains is empty, this will lead to a failure
                if len(current_domain_vals[neighbor]) == 0:
                    return None
                
        # Return the updated domain set
        return current_domain_vals
        
    # Backtrack function
    def backtrack(self, assignment):
        self.nodes += 1
        if self.is_complete(assignment):
            return [assignment, len(assignment)]
        var = self.select_unassigned_var(assignment)
        for value in self.order_domain_values(var):
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                next_domain = self.forward_check(var, assignment)
                if next_domain:
                    # Add the filtered domains to the domain stack
                    self.remaining_domain_stack.append(next_domain)
                    result = self.backtrack(assignment)
                    if result:
                        return result
                    self.remaining_domain_stack.pop()
                del assignment[var]

    # The search
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
            initial_state = [[int(x) for x in line.split()] for line in test_file_content.split("\n") if line.strip()]

            # Yield the initial state matrix and the problem number
            yield initial_state, entry_name.replace('Input', '').replace('.txt', '')

def write_solution(output_file, depth, nodes, assignment):
    board = [[0] * 9 for num in range(9)]
    for (i, j), val in assignment.items():
        board[i][j] = val

    output_file.write(str(depth) + "\n")
    output_file.write(str(nodes) + "\n")
    for i in range(9):
        output_file.write(" ".join(str(board[i][j]) for j in range(9)) + "\n")

def main():
    # For each problem (input file)
    for initial_state, problem_number in parse_input():
        output_file = open(f"{output_directory_path}/Output{problem_number}.txt", "w") # Write to an output file with the naming convention required
        csp = ConstraintSatisfactionProblem(initial_state) # Perform the backtracking search
        result = csp.backtracking_search()
        if not result:
                print("failure")
                output_file.write("0\n")
                output_file.write(str(csp.nodes) + "\n")
                continue
        goal_node, depth = result
        write_solution(output_file, depth, csp.nodes, goal_node)
        output_file.close()

main()