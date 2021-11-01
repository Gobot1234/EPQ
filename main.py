from miner import *
from miner.asteroid import ASTEROIDS, load


def main() -> None:
    load()

    def traveller(asteroid: Asteroid) -> float:
        miner = Miner(Earth)
        miner.travel_to(asteroid)
        carrying = miner.carrying
        miner.travel_to(miner.base_station)

        print("Miner going to", asteroid.identifier, "made", miner.profit, carrying)
        asteroid.miner = miner
        return miner.profit

    # funny bug with the most profit being 0
    sorted_asteroids = sorted(ASTEROIDS, key=traveller)

    print(
        "Most profitable asteroid is", sorted_asteroids[0].identifier, f"making you ${sorted_asteroids[0].miner.profit}"
    )


if __name__ == "__main__":
    main()
