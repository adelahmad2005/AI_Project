from agent_base import Agent
from game import GameState
from evaluation import betterEvaluationFunction  # Add this import

class MinimaxAgent(Agent):
    def get_action(self, state: GameState, depth=None):
        _, action = self.minimax(state, depth_limit=depth, current_depth=0)
        return action

    def minimax(self, state, depth_limit=None, current_depth=0):  # Add current_depth parameter
        """
        Returns: (value, best_action)
        """
        # Terminal state check
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
                value, _ = self.minimax(successor, depth_limit, current_depth + 1)
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            return best_value, best_action
        
        else:  # MIN player ('O')
            best_value = float('inf')
            best_action = None
            
            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self.minimax(successor, depth_limit, current_depth + 1)
                
                if value < best_value:
                    best_value = value
                    best_action = action
            
            return best_value, best_action