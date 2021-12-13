from tabulate import tabulate

from miner import *
from miner.asteroid import ASTEROIDS, load


def main() -> None:
    load()

    def traveller(asteroid: Asteroid) -> float:
        miner = Miner(Moon)
        asteroid.miner = miner
        try:
            miner.travel_to(asteroid)
        except MeanAnomalyFails:
            return -math.inf
        print(miner.time_at_arrival)
        miner.time_at_arrival += timedelta(days=120)  # stay for 4 months to mine then prepare to return
        miner.travel_to(miner.base_station)
        print(miner.time_at_arrival)

        return miner.profit

    sorted_asteroids = sorted(ASTEROIDS, key=traveller, reverse=True)

    table = tabulate(
        (
            (
                a.identifier,
                f"{a.miner.profit:,}",
                ", ".join(c.material.name for c in a.miner.carrying or ()) or "Nothing",
                f"{a.miner.distance_travelled:,}",
                str(a.miner.elapsed_time)[: -len(".000000")],  # strip microseconds
                f"{a.miner.time_at_arrival:%d/%m/%Y %H:%M}",
            )
            for a in sorted_asteroids
        ),
        (
            "Identifier",
            "Profit (USD)",
            "Collected Materials",
            "Distance travelled (m)",
            "Elapsed time",
            "Return date mission (dd-mm-YYYY HH:MM)",
        ),
        tablefmt="grid",
    )
    with open(f"media/{sorted_asteroids[0].miner.base_station.__class__.__name__} Base Table.txt", "w+") as f:
        f.write(table)

    print(table)


if __name__ == "__main__":
    main()
