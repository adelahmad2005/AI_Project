# alphabeta_agent.py
from agent_base import Agent
from game import GameState
from evaluation import betterEvaluationFunction

class AlphaBetaAgent(Agent):
    def get_action(self, state: GameState, depth=None):
        """
        Returns the best move index (0-8) for Tic-Tac-Toe using Alpha-Beta pruning.
        """
        value, action = self.alphabeta(state, alpha=float('-inf'), beta=float('inf'),
                                       depth_limit=depth, current_depth=0)
        # Safety fallback (never return None)
        if action is None:
            legal = state.get_legal_actions()
            if legal:
                action = legal[0]
        return action

    def alphabeta(self, state: GameState, alpha, beta, depth_limit, current_depth):
        """
        Recursive alpha-beta search returning (value, best_action).
        
        TODO: Implement the alpha-beta pruning algorithm here.
        
        The algorithm should:
        1. Check if the state is terminal and return (utility, None) if so
        2. Check if depth limit is reached and use heuristic evaluation if so
        3. For MAX player ('X'): 
           - Find action that maximizes value
           - Update alpha and prune when alpha >= beta
        4. For MIN player ('O'):
           - Find action that minimizes value  
           - Update beta and prune when beta <= alpha
        5. Return the best value and corresponding action as a tuple
        
        Hint: Use betterEvaluationFunction(state) for non-terminal cutoff evaluation
        """
        if state.is_terminal():
            return state.utility(), None
        
        # Depth limit check
        if depth_limit is not None and current_depth >= depth_limit:
            return betterEvaluationFunction(state), None
        
        legal_actions = state.get_legal_actions()
        
        if state.to_move == 'X':  # MAX player
            best_value = float('-inf')
            best_action = None
            
            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self.alphabeta(successor, alpha, beta, depth_limit, current_depth + 1)
                
                if value > best_value:
                    best_value = value
                    best_action = action
                
                # Update alpha and check for pruning
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break  # Beta cutoff
            
            return best_value, best_action
        
        else:  # MIN player ('O')
            best_value = float('inf')
            best_action = None
            
            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self.alphabeta(successor, alpha, beta, depth_limit, current_depth + 1)
                
                if value < best_value:
                    best_value = value
                    best_action = action
                
                # Update beta and check for pruning
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return best_value, best_action
