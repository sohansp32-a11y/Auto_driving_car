import pygame
import math

pygame.init()


# =========================
# SETTINGS
# =========================

WIDTH = 800
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self Driving Car")

clock = pygame.time.Clock()


# =========================
# TRACK
# =========================

track = pygame.image.load("tracks/track.png").convert()


# =========================
# CAR
# =========================

car_x = 100
car_y = 300

car_width = 40
car_height = 70

angle = 0
speed = 0


# =========================
# SENSORS
# =========================

SENSOR_LENGTH = 150

SENSOR_ANGLES = [
    -60,
    -30,
    0,
    30,
    60
]


# =========================
# SENSOR FUNCTION
# =========================

def get_sensor(car_x, car_y, car_angle, sensor_angle):

    # Sensor's final angle
    sensor_direction = math.radians(
        car_angle + sensor_angle
    )

    # Check every pixel along the sensor
    for distance in range(SENSOR_LENGTH):

        sensor_x = int(
            car_x
            - math.sin(sensor_direction) * distance
        )

        sensor_y = int(
            car_y
            - math.cos(sensor_direction) * distance
        )

        # Outside the track
        if (
            sensor_x < 0
            or sensor_x >= WIDTH
            or sensor_y < 0
            or sensor_y >= HEIGHT
        ):
            return distance

        # Get pixel color
        pixel = track.get_at(
            (sensor_x, sensor_y)
        )

        red, green, blue, alpha = pixel

        # White = kill zone
        if (
            red > 200
            and green > 200
            and blue > 200
        ):
            return distance

    print(
        f"Car: ({car_x:.1f}, {car_y:.1f}) | "
        f"Sensors: {sensor_values}"
    )

    # Nothing found within sensor range
    return SENSOR_LENGTH


# =========================
# GAME LOOP
# =========================

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False


    # =========================
    # INPUT
    # =========================

    keys = pygame.key.get_pressed()


    # Acceleration

    if keys[pygame.K_UP]:
        speed += 0.2

    if keys[pygame.K_DOWN]:
        speed -= 0.2


    # =========================
    # FRICTION
    # =========================

    speed *= 0.98


    # =========================
    # STEERING
    # =========================

    if keys[pygame.K_LEFT]:
        angle += 3

    if keys[pygame.K_RIGHT]:
        angle -= 3


    # =========================
    # MOVEMENT
    # =========================

    radians = math.radians(angle)

    car_x -= math.sin(radians) * speed
    car_y -= math.cos(radians) * speed


    # =========================
    # KEEP CAR ON SCREEN
    # =========================

    car_x = max(
        0,
        min(WIDTH - 1, car_x)
    )

    car_y = max(
        0,
        min(HEIGHT - 1, car_y)
    )


    # =========================
    # CHECK ROAD
    # =========================

    pixel = track.get_at(
        (int(car_x), int(car_y))
    )

    red, green, blue, alpha = pixel

    on_road = (
        red < 50
        and green < 50
        and blue < 50
    )


    # =========================
    # CRASH
    # =========================

    if not on_road:

        print("💥 CRASH!")

        car_x = 100
        car_y = 300

        angle = 0
        speed = 0


    # =========================
    # GET SENSOR VALUES
    # =========================

    sensor_values = []

    for sensor_angle in SENSOR_ANGLES:

        distance = get_sensor(
            car_x,
            car_y,
            angle,
            sensor_angle
        )

        sensor_values.append(distance)


    # =========================
    # DRAW
    # =========================

    screen.blit(
        track,
        (0, 0)
    )


    # =========================
    # DRAW SENSORS
    # =========================

    for sensor_angle, distance in zip(
        SENSOR_ANGLES,
        sensor_values
    ):

        sensor_direction = math.radians(
            angle + sensor_angle
        )

        end_x = (
            car_x
            - math.sin(sensor_direction) * distance
        )

        end_y = (
            car_y
            - math.cos(sensor_direction) * distance
        )

        pygame.draw.line(
            screen,
            (255, 0, 0),
            (car_x, car_y),
            (end_x, end_y),
            2
        )


    # =========================
    # CREATE CAR
    # =========================

    car_surface = pygame.Surface(
        (car_width, car_height),
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


    # =========================
    # ROTATE CAR
    # =========================

    rotated_car = pygame.transform.rotate(
        car_surface,
        angle
    )

    car_rect = rotated_car.get_rect(
        center=(car_x, car_y)
    )

    screen.blit(
        rotated_car,
        car_rect
    )


    pygame.display.flip()

    clock.tick(60)


pygame.quit()