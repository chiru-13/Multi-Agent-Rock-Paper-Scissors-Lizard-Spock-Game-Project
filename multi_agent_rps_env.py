import pygame
import pygame_gui
import gym
from gym import spaces
import numpy as np

#replace the image path with the loction of the image in the device u are running
rock_image = pygame.image.load(r"C:\Users\saisr\Desktop\AI_project\Images\rock.png")
paper_image = pygame.image.load(r"C:\Users\saisr\Desktop\AI_project\Images\paper.png")
scissors_image = pygame.image.load(r"C:\Users\saisr\Desktop\AI_project\Images\scissor.png")
lizard_image = pygame.image.load(r"C:\Users\saisr\Desktop\AI_project\Images\lizard.png")
spock_image = pygame.image.load(r"C:\Users\saisr\Desktop\AI_project\Images\spock.png")

image_width = 150
image_height = 150

rock_image = pygame.transform.scale(rock_image, (image_width, image_height))
paper_image = pygame.transform.scale(paper_image, (image_width, image_height))
scissors_image = pygame.transform.scale(scissors_image, (image_width, image_height))
lizard_image = pygame.transform.scale(lizard_image, (image_width, image_height))
spock_image = pygame.transform.scale(spock_image, (image_width, image_height))


pygame.mixer.init()
pygame.mixer.music.load(r"C:\Users\saisr\Downloads\password-infinity-123276.mp3")  # Replace with the actual path to your music file in the device u are running
pygame.mixer.music.play(-1)  # -1 makes the music loop indefinitely

class MultiAgentRockPaperScissorsEnv(gym.Env):
    dark_mode = False  # Class variable for dark_mode

    def __init__(self, num_actions=5):
        super(MultiAgentRockPaperScissorsEnv, self).__init__()
        self.num_actions = num_actions
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(1)
        self.state = 0
        self.max_cycles = 10
        self.current_cycle = 0
        self.players = 2
        self.agents = [self.make_agent() for _ in range(self.players)]
        self.screen_width, self.screen_height = 900, 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Rock Paper Scissors Lizard Spock")
        self.image_size = 150
        self.clock = pygame.time.Clock()
        self.action_history = []
        self.agent_names = ["Agent 1", "Agent 2"]
        self.agent_scores = [0, 0]
        self.gui_manager = pygame_gui.UIManager((self.screen_width, self.screen_height))
        self.dark_light_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((self.screen_width - 100, 10), (90, 30)),
            text='Dark Mode',
            manager=self.gui_manager
        )

    def make_agent(self):
        return {
            "action": None,
            "reward": 0
        }

    def step(self, actions):
        assert len(actions) == self.players, "Each agent must select an action."
        for i in range(self.players):
            self.agents[i]["action"] = actions[i]

        self.action_history.append([agent["action"] for agent in self.agents])

        winner = self.determine_winner(self.agents[0]["action"], self.agents[1]["action"])
        if winner is not None:
            self.agent_scores[winner] += 1

        for i in range(self.players):
            self.agents[i]["reward"] = 1 if i == winner else 0

        self.state = 0
        self.current_cycle += 1
        done = self.current_cycle >= self.max_cycles

        self.render()
        self.clock.tick(2)
        pygame.display.flip()

        # Handle events including button press
        events = pygame.event.get()
        for event in events:
            self.gui_manager.process_events(event)
        self.handle_events(events)

        return self.state, [agent["reward"] for agent in self.agents], done, {}

    def reset(self):
        for i in range(self.players):
            self.agents[i]["action"] = None
            self.agents[i]["reward"] = 0
        self.state = 0
        self.current_cycle = 0
        self.action_history = []
        self.agent_scores = [0, 0]
        return self.state

    def render(self, mode='human'):
        if MultiAgentRockPaperScissorsEnv.dark_mode:
            self.screen.fill((0, 0, 0))
        else:
            self.screen.fill((255, 255, 255))

        font = pygame.font.Font(None, 36)
        for i in range(self.players):
            text = font.render(f"{self.agent_names[i]}", True, (255, 255, 255) if MultiAgentRockPaperScissorsEnv.dark_mode else (0, 0, 0))
            self.screen.blit(text, (self.screen_width // 2.5 - 120 + i * 300, 560))

        offset = 0
        for actions in reversed(self.action_history):
            if len(actions) == self.players:
                for i in range(self.players):
                    x_pos = self.screen_width * (i + 1) // (self.players + 1) - self.image_size // 2
                    y_pos = self.screen_height // 2 - self.image_size // 2 - offset
                    self.draw_image(actions[i], x_pos, y_pos)
                offset += 160

        for i in range(self.players):
            x_pos = self.screen_width * (i + 1) // (self.players + 1) - self.image_size // 2
            y_pos = self.screen_height // 2 + self.image_size // 2 + 20
            self.draw_image(self.agents[i]["action"], x_pos, y_pos)

        self.gui_manager.update(1 / 60.0)
        self.gui_manager.draw_ui(self.screen)
        pygame.display.flip()

    def draw_image(self, action, x_pos, y_pos):
        if action == 0:
            self.screen.blit(rock_image, (x_pos, y_pos))
        elif action == 1:
            self.screen.blit(paper_image, (x_pos, y_pos))
        elif action == 2:
            self.screen.blit(scissors_image, (x_pos, y_pos))
        elif action == 3:
            self.screen.blit(lizard_image, (x_pos, y_pos))
        elif action == 4:
            self.screen.blit(spock_image, (x_pos, y_pos))

    def close(self):
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        pygame.quit()

    def determine_winner(self, action1, action2):
        if (action1 - action2) % self.num_actions == 0:
            return None
        if (action1 - action2) % self.num_actions in [1, 3]:
            return 0
        else:
            return 1

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.dark_light_button:
                        self.toggle_dark_light_mode()

    def toggle_dark_light_mode(self):
        MultiAgentRockPaperScissorsEnv.dark_mode = not MultiAgentRockPaperScissorsEnv.dark_mode

        if MultiAgentRockPaperScissorsEnv.dark_mode:
            self.screen.fill((0, 0, 0))
            self.dark_light_button.set_text('Light Mode')
        else:
            self.screen.fill((255, 255, 255))
            self.dark_light_button.set_text('Dark Mode')