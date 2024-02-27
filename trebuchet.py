# Start -> 14:50, end -> 15:50
# Start -> 16:05
import math
import random
from typing import List, Tuple
import matplotlib.pyplot as plt


VELOCITY = 50
HEIGHT = 100
GRAVITY = 9.81


class Radians:
    def __init__(self, degrees: int) -> None:
        self.degrees: float = math.radians(degrees)

    def __float__(self):
        return self.degrees


def get_x(angle: Radians, t: float) -> float:
    return VELOCITY * math.cos(angle) * t


def get_y(angle: Radians, t: float) -> float:
    return HEIGHT + (VELOCITY * math.sin(angle) * t) - (0.5 * GRAVITY * t ** 2)


def distance(angle: Radians) -> float:
    sin = math.sin(angle)
    cos = math.cos(angle)
    vo_sin = VELOCITY * sin
    vo_cos = VELOCITY * cos
    sqrt = math.sqrt((vo_sin**2) + (2 * GRAVITY * HEIGHT))
    div = (vo_sin + sqrt) / GRAVITY
    return vo_cos * div


def plot_distance(angle: Radians, d: float) -> Tuple[List[float], List[float]]:
    x_pos: List[float] = []
    y_pos: List[float] = []

    for t in range(0, int(d)*100):
        x = get_x(angle, t/100)
        y = get_y(angle, t/100)
        if y < 0:
            break
        x_pos.append(x)
        y_pos.append(y)

    return (x_pos, y_pos)


def spawn_target() -> float:
    return random.uniform(50.0, 340.0)


def check_hit(distance: float, target: float) -> bool:
    low, high = target - 5, target + 5
    return low <= distance <= high


if __name__ == "__main__":
    target = spawn_target()
    tries = 0
    while True:
        tries += 1
        print(f"the target is at: {target}")
        angle = Radians(int(input("angle? (degrees): ")))
        d = distance(angle)
        print(d)
        if not check_hit(d, target):
            continue
        x, y = plot_distance(angle, d)
        for i, _ in enumerate(x):
            print(f"{x[i]:.2f} {y[i]:.2f}")
        plt.plot(x, y, label="pocisk z Warwolf", color="blue")
        plt.axvline(
            x=d,
            color='red',
            linestyle='--',
            label=f"Distance: {d:.2f}"
        )
        plt.grid()
        plt.xlabel("distance")
        plt.ylabel("height")
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.show()
        break
