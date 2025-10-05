def is_valid(state):
    return 0 <= state.index('.') < len(state)

def get_successors(state):
    successors = []
    index = state.index('.')
    moves = [1, 2]
    for move in moves:
        if index + move < len(state):
            if state[index + move] == 'W':
                new_state = list(state)
                new_state[index], new_state[index + move] = new_state[index + move], new_state[index]
                new_state = ''.join(new_state)
                if is_valid(new_state):
                    successors.append(new_state)
    for move in moves:
        if index - move >= 0:
            if state[index - move] == 'E':
                new_state = list(state)
                new_state[index], new_state[index - move] = new_state[index - move], new_state[index]
                new_state = ''.join(new_state)
                if is_valid(new_state):
                    successors.append(new_state)
    return successors
def dfs_recursive(state, goal_state, path, visited):
    if state in visited:
        return None
    visited.add(state)
    path.append(state)

    if state == goal_state:
        return path

    for successor in get_successors(state):
        result = dfs_recursive(successor, goal_state, path, visited)
        if result is not None:
            return result

    path.pop()  # Backtrack
    return None

def dfs(start_state, goal_state):
    visited = set()
    path = []
    return dfs_recursive(start_state, goal_state, path, visited)

start_state = "EEE.WWW"
goal_state = "WWW.EEE"

dfs_solution = dfs(start_state, goal_state)
print("\nDFS Solution:")
if dfs_solution:
    for step in dfs_solution:
        print(step)
else:
    print("No solution found.")