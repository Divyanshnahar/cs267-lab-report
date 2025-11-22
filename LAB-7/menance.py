import random
import math

class MENACE:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.matchboxes = {}
        self.moves_made = []
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def get_move(self, board):
        if board not in self.matchboxes:
            self.matchboxes[board] = {i: 0 for i in range(9) if board[i] == '-'}
        
        moves = self.matchboxes[board]
        
        if random.random() < self.exploration_rate:
            move = random.choice(list(moves.keys()))
            print(f"Exploring: Chose move {move}")
        else:
            move = max(moves, key=moves.get)
            print(f"Exploiting: Chose move {move} with Q-value {moves[move]:.2f}")
        
        self.moves_made.append((board, move))
        return move

    def learn(self, final_reward):
        for i in range(len(self.moves_made) - 1, -1, -1):
            board, move = self.moves_made[i]
            next_board = self.moves_made[i+1][0] if i+1 < len(self.moves_made) else None
            
            if next_board:
                next_max_q = max(self.matchboxes[next_board].values()) if self.matchboxes[next_board] else 0
                reward = self.calculate_reward(board, move)
                new_q = (1 - self.learning_rate) * self.matchboxes[board][move] + \
                        self.learning_rate * (reward + self.discount_factor * next_max_q)
            else:
                new_q = final_reward
            
            self.matchboxes[board][move] = new_q
            print(f"Updated Q-value for move {move} on board {board}: {new_q:.2f}")
        
        self.moves_made = []

    def calculate_reward(self, board, move):
        new_board = board[:move] + 'X' + board[move+1:]
        if check_win(new_board):
            return 1  
        elif check_win(new_board.replace('X', 'O').replace('-', 'X')):
            return 0.5  
        else:
            return 0.1 

def check_win(board):
    winning_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  
        (0, 3, 6), (1, 4, 7), (2, 5, 8), 
        (0, 4, 8), (2, 4, 6)  
    ]
    for a, b, c in winning_combinations:
        if board[a] == board[b] == board[c] != '-':
            return True
    return False

def play_game(menace, opponent):
    board = '-' * 9
    current_player = menace

    while True:
        print(f"\nCurrent board:\n{board[:3]}\n{board[3:6]}\n{board[6:]}")
        move = current_player.get_move(board)
        board = board[:move] + ('X' if current_player == menace else 'O') + board[move+1:]

        if check_win(board):
            return 1 if current_player == menace else -1
        if '-' not in board:
            return 0

        current_player = opponent if current_player == menace else menace

class RandomPlayer:
    def get_move(self, board):
        return random.choice([i for i in range(9) if board[i] == '-'])


menace = MENACE()
opponent = RandomPlayer()

print("Training MENACE...")
for episode in range(10000):
    result = play_game(menace, opponent)
    menace.learn(result)
    if episode % 1000 == 0:
        print(f"Episode {episode} completed")

print("\nTraining completed!")

board = '-' * 9
current_player = menace

print("\nLet's play a game against MENACE!")
while True:
    print(f"\nCurrent board:\n{board[:3]}\n{board[3:6]}\n{board[6:]}")
    
    if current_player == menace:
        move = menace.get_move(board)
        print(f"MENACE plays: {move}")
    else:
        move = int(input("Your move (0-8): "))
        while board[move] != '-':
            move = int(input("Invalid move. Try again (0-8): "))

    board = board[:move] + ('X' if current_player == menace else 'O') + board[move+1:]

    if check_win(board):
        print(f"\nFinal board:\n{board[:3]}\n{board[3:6]}\n{board[6:]}")
        print("MENACE wins!" if current_player == menace else "You win!")
        break
    if '-' not in board:
        print(f"\nFinal board:\n{board[:3]}\n{board[3:6]}\n{board[6:]}")
        print("It's a draw!")
        break

    current_player = opponent if current_player == menace else menace


if current_player == menace:
    menace.learn(-1) 
else:
    menace.learn(1)  