class RewardMachine:
    def __init__(self):
        # Define RM states, events, and transitions
        self.states = ["default", "emergency_priority", "pedestrian_crossing", "queue_minimization"]
        self.initial_state = "default"
        self.current_state = self.initial_state
        self.transitions = {
            "default": {
                "emergency_detected": "emergency_priority",
                "pedestrian_button_pressed": "pedestrian_crossing",
                "queue_threshold_exceeded": "queue_minimization"
            },
            "emergency_priority": {"emergency_cleared": "default"},
            "pedestrian_crossing": {"pedestrian_cleared": "default"},
            "queue_minimization": {"queue_cleared": "default"}
        }
        self.rewards = {
            ("default", "vehicle_arrival"): -1,  # Penalize queue buildup
            # ("emergency_priority", "emergency_cleared"): 10,  # Reward clearing emergency
            ("pedestrian_crossing", "pedestrian_cleared"): 5,  # Reward pedestrian clearance
            ("queue_minimization", "queue_cleared"): 2  # Reward queue reduction
        }

    def get_next_state(self, event):
        """Get the next state based on the current state and event."""
        return self.transitions.get(self.current_state, {}).get(event, self.current_state)

    def get_reward(self, event):
        """Get the reward for a state-event pair."""
        return self.rewards.get((self.current_state, event), 0)  # Default reward is 0

    def update_state(self, event):
        """Update the RM's state based on the event."""
        self.current_state = self.get_next_state(event)
