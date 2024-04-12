from decimal import Decimal


def arbitrage(price_matrix, start, steps):
    best_rate, best_path = 0, []

    def dfs(u, rate, path):
        nonlocal best_rate, best_path

        if len(path) == steps + 1:
            if u == start and rate > best_rate:
                best_rate = rate
                best_path = path.copy()
            return

        for v in range(len(price_matrix)):
            if v == u:
                continue
            path.append(v)
            dfs(v, rate * price_matrix[u][v], path)
            path.pop()

    dfs(start, 1, [start])
    return best_rate, best_path


def main():
    price_matrix = [
        [Decimal("1.00"), Decimal("0.48"), Decimal("1.52"), Decimal("0.71")],
        [Decimal("2.05"), Decimal("1.00"), Decimal("3.26"), Decimal("1.56")],
        [Decimal("0.64"), Decimal("0.30"), Decimal("1.00"), Decimal("0.46")],
        [Decimal("1.41"), Decimal("0.61"), Decimal("2.08"), Decimal("1.00")],
    ]
    seashells = 3
    steps = 5
    print(arbitrage(price_matrix, seashells, steps))


if __name__ == "__main__":
    main()
