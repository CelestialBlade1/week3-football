import random

class Transition:
    state: tuple
    action: int
    p: float
    q: float
    new_state: list

    # Initializing all the required variables
    def __init__(self, state: tuple, action: int, skill_p: float, skill_q: float, pos_R_initial: int):
        self.state = state
        self.changeable_state = list(state)
        self.action = action
        self.p = skill_p
        self.q = skill_q
        self.pos_R_in = pos_R_initial
        self.end = 1 # Variable to keep track the situation of the game

    # Function to determine the consequence of action and the corresponding reward obtained
    def get_transition(self) -> tuple:
        if self.action < 8:
            state_val = self.movement(self.changeable_state, self.action)
        elif self.action == 8:
            state_val = self.passing(self.changeable_state, self.action)
        elif self.action == 9:
            state_val = self.shooting(self.changeable_state, self.action)

        reward = self.get_reward()
        state_val = tuple(state_val)
        return state_val, reward

    def get_reward(self) -> int:
        if self.end == 0:
            reward = -10
        elif self.end == 2:
            reward = 10
        else:
            reward = -1
        return reward

    def movement(self, s: list, a: int) -> list:
        # Determine movement based on action
        if a in [0, 1, 2, 3]:
            # Player 1 movement
            if s[3] == 1:
                flag_0 = self.check_tackle_possibility(s, a)
                success_prob = 0.5 - self.p if flag_0 else 1 - 2 * self.p
            else:
                # Player 2 movement
                success_prob = 1 - self.p

            move_offsets = [(-1, 0), (1, 0), (-4, 0), (4, 0)]
            dx, dy = move_offsets[a]

            success = random.choices([True, False], [success_prob, 1 - success_prob])[0]
            # Chose the action performed based on the success
            if success:
                s[0] += dx
                s[1] += dy
                if self.check_within_bounds(s[0]):
                    s[0] -= dx
                    s[1] -= dy
                    self.end = 0
            else:
                self.end = 0

        elif a in [4, 5, 6, 7]:
            if s[3] == 2:
                flag_0 = self.check_tackle_possibility(s, a)
                success_prob = 0.5 - self.p if flag_0 else 1 - 2 * self.p
            else:
                success_prob = 1 - self.p

            move_offsets = [(0, -1), (0, 1), (0, -4), (0, 4)]
            dx, dy = move_offsets[a - 4]

            success = random.choices([True, False], [success_prob, 1 - success_prob])[0]
            if success:
                s[0] += dx
                s[1] += dy
                if self.check_within_bounds(s[1]):
                    s[0] -= dx
                    s[1] -= dy
                    self.end = 0
            else:
                self.end = 0

        return s

    def check_tackle_possibility(self, s: list, a: int) -> bool:
        move_offsets = [(-1, 0), (1, 0), (-4, 0), (4, 0), (0, -1), (0, 1), (0, -4), (0, 4)]
        # Selecting moves based on the action
        dx, dy = move_offsets[a]
        x1 = s[0] + dx
        y1 = s[1] + dy
        # Returning a bool value based on the action taken and the position of R
        if a < 4:
            return x1 == s[2] or (x1 == self.pos_R_in and s[0] == s[2])
        
        else:
            return y1 == s[2] or (y1 == self.pos_R_in and s[1] == s[2])

    def check_within_bounds(self, x: int) -> bool:
        return x < 1 or x > 16

    def passing(self, s: list, a: int) -> list:
        coords = [(i, j) for i in range(4) for j in range(4)]
        B1_coords = coords[s[0] - 1]
        B2_coords = coords[s[1] - 1]
        R_coords = coords[s[2] - 1]
        # Checking the probability for success depending upon the  given condition
        x = 0.5 * self.q - 0.05 * max(abs(B1_coords[0]-B2_coords[0]), abs(B1_coords[1]-B2_coords[1]))

        flag = self.check_defender(B1_coords, B2_coords, R_coords)

        # Adjusting the success probability based on the presence of defender
        if flag:
            success_prob = x 
        else:
            success_prob = 2 * x 

        success = random.choices([True, False], [success_prob, 1 - success_prob])[0]
        s[3] = 2 if s[3] == 1 else 1
        if not success:
            self.end = 0

        return s
    # Function checking collinearity to check for the presence of defender
    def check_defender(self, a: tuple, b: tuple, c: tuple) -> bool:
        area = a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])
        return area == 0

    # Funciton to check whether a goal is being scored
    def shooting(self, s: list, a: int) -> list:
        coords = [(i, j) for i in range(4) for j in range(4)]
        B1_coords = coords[s[0] - 1]
        B2_coords = coords[s[1] - 1]
        R_coords = coords[s[2] - 1]

        flag = self.check_goalie(R_coords)

        # Adjusting the success probability based on the presence or absence of goalkeeper
        if flag:
            if s[3] == 1:
                x = 0.5 * (self.q - 0.2 * (3 - B1_coords[0]))
            else:
                x = 0.5 * (self.q - 0.2 * (3 - B2_coords[0]))
        else:
            if s[3] == 1:
                x = self.q - 0.2 * (3 - B1_coords[0])
            else:
                x = self.q - 0.2 * (3 - B2_coords[0])

        success = random.choices([True, False], [x, 1 - x])[0]
        self.end = 2 if success else 0

        return s

    # Function to check the presence of a goal keeper
    def check_goalie(self, r: tuple) -> bool:
        return r in [(1, 3), (2, 3)]
