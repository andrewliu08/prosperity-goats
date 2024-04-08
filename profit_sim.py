import json
import random
import statistics


def generate_fish_value():
    """
    Generate a value in [900, 1000] with probability scaling
    linearly.
    """
    # PDF: f(x) = x/5000
    # CDF: F(x) = x^2/10000
    # Inverse CDF: F^-1(x) = 100*sqrt(x)
    p = random.random()
    val = 100 * (p ** 0.5)
    return 900 + val


def test_generate_fish_value():
    random.seed(0)
    iterations = 10000
    vals = [generate_fish_value() for _ in range(iterations)]
    assert 900 <= min(vals) <= 1000
    assert 900 <= max(vals) <= 1000

    buckets = [0] * 10
    for val in vals:
        bucket = int((val - 900) // 10)
        buckets[bucket] += 1
    
    actual_cdf = 0.0
    for i in range(10):
        fraction = buckets[i] / iterations
        actual_cdf += fraction
        expected_cdf = ((i + 1) * 10) ** 2 / 10000 
        print(f"Fraction: {fraction}, Actual CDF: {actual_cdf}, Expected CDF: {expected_cdf}")


def profit_ev_and_std(a, b, iterations):
    fish_vals = [generate_fish_value() for _ in range(iterations)]

    profits = []
    for val in fish_vals:
        if val < a:
            profits.append(1000 - a)
        elif val < b:
            profits.append(1000 - b)
        else:
            profits.append(0)
    
    ev = sum(profits) / iterations
    stdev = statistics.stdev(profits)
    return ev, stdev


def simulate():
    """
    Calculate the expected value, standard deviation, and Sharpe ratio for
    a given strategy of bidding at price A B.
    """
    a_range = range(900, 1001, 1)
    b_range = range(900, 1001, 1)

    data = []
    for a in a_range:
        for b in b_range:
            if a >= b:
                continue
            ev, std = profit_ev_and_std(a, b, 1000000)
            std = max(std, 0.0001)
            sharpe = ev / std
            print(f"A: {a}, B: {b}, EV: {ev}, STD: {std}, Sharpe: {sharpe}")
            data.append({"A": a, "B": b, "EV": ev, "STD": std, "Sharpe": sharpe})

    with open("profit_sim.json", "w") as f:
        json.dump(data, f)


def best_value():
    with open("profit_sim.json", "r") as f:
        data = json.load(f)
    
    ev_threshold = 19

    best_a = None
    best_b = None
    best_ev = None
    best_std = None
    best_sharpe = None
    for item in data:
        if item["EV"] < ev_threshold:
            continue

        # if best_ev is None or item["EV"] > best_ev:
        if best_sharpe is None or item["Sharpe"] > best_sharpe:
            best_a = item["A"]
            best_b = item["B"]
            best_ev = item["EV"]
            best_std = item["STD"]
            best_sharpe = item["Sharpe"]
    
    print(f"Best EV: {best_ev}, std: {best_std}, sharpe: {best_sharpe}, A: {best_a}, B: {best_b}")


if __name__ == "__main__":
    # simulate()
    best_value()
    # test_generate_fish_value()
