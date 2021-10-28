import math

from miner import *
from miner.asteroid import ASTEROIDS, load


def main() -> None:
    load()

    winner: tuple[float, Asteroid] = (-math.inf, None)  # type: ignore  # funny bug with the most profit being 0
    for asteroid in ASTEROIDS:
        miner = Miner(Earth)
        miner.travel_to(asteroid)
        miner.travel_to(miner.base_station)
        if winner[0] < miner.profit:
            winner = (miner.profit, asteroid)

    print("Most profitable asteroid is", winner[1], f"making you ${winner[0]}")


if __name__ == "__main__":
    main()
