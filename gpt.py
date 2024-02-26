import math
from typing import Callable, Mapping
from datetime import datetime, timedelta

def get_name() -> str:
    return input("What is your name?: ")

def get_year() -> int:
    while True:
        try:
            from_user = input("What year were you born?: ")
            return int(from_user)
        except ValueError:
            print(f"{from_user} is not a number!")

def get_month() -> int:
    while True:
        try:
            from_user = input("What month were you born?[1-12]: ")
            month = int(from_user)
            if 1 <= month <= 12:
                return month
        except ValueError:
            print(f"{from_user} is not a number!")

def get_day() -> int:
    while True:
        try:
            from_user = input("What day were you born?[1-31]: ")
            day = int(from_user)
            if 1 <= day <= 31:
                return day
        except ValueError:
            print(f"{from_user} is not a number!")

def biorythm(delta: Callable[[], timedelta]) -> Mapping[str, float]:
    t = delta().days
    physical = math.sin((2 * math.pi / 23) * t)
    emotional = math.sin((2 * math.pi / 28) * t)
    intellectual = math.sin((2 * math.pi / 33) * t)
    return {
        "physical": physical,
        "emotional": emotional,
        "intellectual": intellectual,
    }

def get_info():
    return get_name(), get_year(), get_month(), get_day()

if __name__ == "__main__":
    name, year, month, day = get_info()
    date_of_birth = datetime(year, month, day)

    today = biorythm(lambda: date_of_birth - datetime.now())
    tomorrow = biorythm(lambda: date_of_birth - (datetime.now() + timedelta(1)))

    print("Today's waves are:")
    for k in today:
        print()
        print(f"{k}: {today[k]}")
        if today[k] < -0.5:
            print(f"I am sorry you are having a bad {k} wave today, {name}!")
            if today[k] < tomorrow[k]:
                print("Tomorrow it will be better!")
        elif today[k] > 0.5:
            print(f"I must congratulate you on having such a great {k} wave today, {name}!")
