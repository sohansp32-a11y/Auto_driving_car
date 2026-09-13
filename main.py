import pygame
import math
import requests
import random
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F

from brain import model


# =========================================================
# DEVICE
# =========================================================

device = "cpu"


# =========================================================
# DQN SETUP
# =========================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

target_model = copy.deepcopy(model)

target_model.eval()


# =========================================================
# REPLAY BUFFER
# =========================================================

class ReplayBuffer:

    def __init__(self, capacity=10000):

        self.buffer = []
        self.capacity = capacity

    def add(self, experience):

        self.buffer.append(experience)

        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):

        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):

        return len(self.buffer)


# IMPORTANT:
# Create this ONCE, outside the game loop

memory = ReplayBuffer(10000)


# =========================================================
# TRAINING
# =========================================================

def train_step(
    batch_size=64,
    gamma=0.99
):

    # Not enough experiences yet
    if len(memory) < batch_size:
        return None

    batch = memory.sample(batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)


    # -----------------------------------------------------
    # CONVERT TO TENSORS
    # -----------------------------------------------------

    states = torch.tensor(
        states,
        dtype=torch.float32,
        device=device
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long,
        device=device
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    next_states = torch.tensor(
        next_states,
        dtype=torch.float32,
        device=device
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=device
    )


    # -----------------------------------------------------
    # CURRENT Q VALUES
    # -----------------------------------------------------

    q_values = model(states)

    current_q = q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)


    # -----------------------------------------------------
    # TARGET Q VALUES
    # -----------------------------------------------------

    with torch.no_grad():

        next_q_values = target_model(next_states)

        max_next_q = next_q_values.max(
            dim=1
        )[0]

        target_q = (
            rewards
            + gamma * max_next_q * (1 - dones)
        )


    # -----------------------------------------------------
    # LOSS
    # -----------------------------------------------------

    loss = F.mse_loss(
        current_q,
        target_q
    )


    # -----------------------------------------------------
    # BACKPROPAGATION
    # -----------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    return loss.item()


# =========================================================
# PYGAME
# =========================================================

pygame.init()


# =========================================================
# SETTINGS
# =========================================================

WIDTH = 800
HEIGHT = 600

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Self Driving Car"
)

clock = pygame.time.Clock()


# =========================================================
# TRACK
# =========================================================

track = pygame.image.load(
    "tracks/track.png"
).convert()


# =========================================================
# CAR
# =========================================================

car_x = 100
car_y = 300

car_width = 40
car_height = 70

angle = 0
speed = 0

crash = False


# =========================================================
# RESET CAR
# =========================================================

def reset_car():

    global car_x
    global car_y
    global angle
    global speed
    global crash

    car_x = 100
    car_y = 300

    angle = 0
    speed = 0

    crash = False


# =========================================================
# ACTION
# =========================================================

def perform_action(action):

    global speed
    global angle

    if action == 0:

        # accelerate
        speed += 0.2

    elif action == 1:

        # turn left
        angle += 3

    elif action == 2:

        # turn right
        angle -= 3


# =========================================================
# REWARD
# =========================================================

def get_reward() -> float:

    if crash:

        return -10

    return 0.5


# =========================================================
# DONE
# =========================================================

def is_done() -> bool:

    if crash:

        return True

    return False


# =========================================================
# SENSORS
# =========================================================

SENSOR_LENGTH = 150

SENSOR_ANGLES = [
    -60,
    -30,
    0,
    30,
    60
]


# =========================================================
# SENSOR FUNCTION
# =========================================================

def get_sensor(
    car_x,
    car_y,
    car_angle,
    sensor_angle
):

    sensor_direction = math.radians(
        car_angle + sensor_angle
    )

    for distance in range(
        SENSOR_LENGTH
    ):

        sensor_x = int(
            car_x
            - math.sin(sensor_direction)
            * distance
        )

        sensor_y = int(
            car_y
            - math.cos(sensor_direction)
            * distance
        )


        # Outside screen

        if (
            sensor_x < 0
            or sensor_x >= WIDTH
            or sensor_y < 0
            or sensor_y >= HEIGHT
        ):

            return distance


        # Get pixel

        pixel = track.get_at(
            (sensor_x, sensor_y)
        )

        red, green, blue, alpha = pixel


        # White = wall / kill zone

        if (
            red > 200
            and green > 200
            and blue > 200
        ):

            return distance


    return SENSOR_LENGTH


# =========================================================
# GET STATE
# =========================================================

def get_state() -> list[float]:

    # IMPORTANT:
    # New list every time

    sensor_values = []

    for sensor_angle in SENSOR_ANGLES:

        distance = get_sensor(
            car_x,
            car_y,
            angle,
            sensor_angle
        )

        sensor_values.append(
            distance
        )

    return sensor_values


# =========================================================
# API PREDICTION
# =========================================================

def get_prediction(state):

    res = requests.post(
        "http://127.0.0.1:8000",
        json={
            "state": state
        }
    )

    res.raise_for_status()

    data = res.json()

    return (
        data["action"],
        data["state"]
    )


# =========================================================
# GAME LOOP
# =========================================================

running = True

while running:

    # -----------------------------------------------------
    # EVENTS
    # -----------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False


    # -----------------------------------------------------
    # CURRENT STATE
    # -----------------------------------------------------

    state = get_state()


    # -----------------------------------------------------
    # AI CHOOSES ACTION
    # -----------------------------------------------------

    prev_state = state.copy()

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        q_values = model(state_tensor)

    action = torch.argmax(q_values).item()


    # -----------------------------------------------------
    # PERFORM ACTION
    # -----------------------------------------------------

    perform_action(action)


    # -----------------------------------------------------
    # MOVE CAR
    # -----------------------------------------------------

    speed *= 0.98

    radians = math.radians(angle)

    car_x -= (
        math.sin(radians)
        * speed
    )

    car_y -= (
        math.cos(radians)
        * speed
    )


    # -----------------------------------------------------
    # KEEP CAR ON SCREEN
    # -----------------------------------------------------

    car_x = max(
        0,
        min(WIDTH - 1, car_x)
    )

    car_y = max(
        0,
        min(HEIGHT - 1, car_y)
    )


    # -----------------------------------------------------
    # CHECK ROAD
    # -----------------------------------------------------

    pixel = track.get_at(
        (
            int(car_x),
            int(car_y)
        )
    )

    red, green, blue, alpha = pixel

    on_road = (
        red < 50
        and green < 50
        and blue < 50
    )


    # -----------------------------------------------------
    # CRASH
    # -----------------------------------------------------

    if not on_road:

        print("💥 CRASH!")

        crash = True


    # -----------------------------------------------------
    # NEXT STATE
    # -----------------------------------------------------

    next_state = get_state()


    # -----------------------------------------------------
    # REWARD
    # -----------------------------------------------------

    reward = get_reward()


    # -----------------------------------------------------
    # DONE
    # -----------------------------------------------------

    done = is_done()


    # -----------------------------------------------------
    # STORE EXPERIENCE
    # -----------------------------------------------------

    memory.add(
        (
            prev_state,
            action,
            reward,
            next_state,
            done
        )
    )


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    loss = train_step()

    if loss is not None:

        print(
            "Loss:",
            loss
        )


    # -----------------------------------------------------
    # RESET AFTER CRASH
    # -----------------------------------------------------

    if done:

        reset_car()


    # =====================================================
    # DRAW
    # =====================================================

    screen.blit(
        track,
        (0, 0)
    )


    # -----------------------------------------------------
    # DRAW SENSORS
    # -----------------------------------------------------

    sensor_values = get_state()

    for sensor_angle, distance in zip(
        SENSOR_ANGLES,
        sensor_values
    ):

        sensor_direction = math.radians(
            angle + sensor_angle
        )

        end_x = (
            car_x
            - math.sin(sensor_direction)
            * distance
        )

        end_y = (
            car_y
            - math.cos(sensor_direction)
            * distance
        )

        pygame.draw.line(
            screen,
            (255, 0, 0),
            (car_x, car_y),
            (end_x, end_y),
            2
        )


    # -----------------------------------------------------
    # CREATE CAR
    # -----------------------------------------------------

    car_surface = pygame.Surface(
        (
            car_width,
            car_height
        ),
        pygame.SRCALPHA
    )


    # Car body

    pygame.draw.rect(
        car_surface,
        (0, 150, 255),
        (
            5,
            5,
            car_width - 10,
            car_height - 10
        ),
        border_radius=8
    )


    # Windshield

    pygame.draw.rect(
        car_surface,
        (40, 40, 40),
        (
            9,
            12,
            car_width - 18,
            18
        ),
        border_radius=4
    )


    # Front marker

    pygame.draw.rect(
        car_surface,
        (255, 220, 0),
        (
            12,
            3,
            car_width - 24,
            6
        ),
        border_radius=2
    )


    # -----------------------------------------------------
    # ROTATE CAR
    # -----------------------------------------------------

    rotated_car = pygame.transform.rotate(
        car_surface,
        angle
    )

    car_rect = rotated_car.get_rect(
        center=(
            car_x,
            car_y
        )
    )


    screen.blit(
        rotated_car,
        car_rect
    )


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    pygame.display.flip()

    clock.tick(60)


# =========================================================
# EXIT
# =========================================================

pygame.quit()