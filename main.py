from miner import *
from miner.asteroid import ASTEROIDS, load
from tabulate import tabulate


def main() -> None:
    load()

    def traveller(asteroid: Asteroid) -> float:
        miner = Miner(Earth)
        try:
            miner.travel_to(asteroid)
        except MeanAnomalyFails:
            return -math.inf
        miner.travel_to(miner.base_station)

        asteroid.miner = miner
        return miner.profit

    # funny bug with the most profit being 0
    sorted_asteroids = sorted(ASTEROIDS, key=traveller)
    print(len(sorted_asteroids))
    print(sorted_asteroids[0].miner.fuel)
    print(
        "Most profitable asteroid is", sorted_asteroids[0].identifier, f"making you ${sorted_asteroids[0].miner.profit}"
    )

    print(
        tabulate(
            (
                (
                    a.identifier,
                    a.miner.profit,
                    ", ".join(c.material.name for c in a.miner.carrying or ()) or "Nothing",
                    None,
                )
                for a in sorted_asteroids
            ),
            ["identifier", "profit", "contents", "time for mission"],
            tablefmt="grid",
        )
    )


if __name__ == "__main__":
    main()
