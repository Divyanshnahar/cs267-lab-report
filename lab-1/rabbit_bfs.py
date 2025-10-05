from collections import deque

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

def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    while queue:
        state, path = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
    return None

start_state = "EEE.WWW"
goal_state = "WWW.EEE"

bfs_solution = bfs(start_state, goal_state)
print("BFS Solution:")
if bfs_solution:
    for step in bfs_solution:
        print(step)
else:
    print("No solution found.")