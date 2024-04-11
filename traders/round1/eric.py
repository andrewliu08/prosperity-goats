import json
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
import statistics
from typing import Any

SEASHELLS = "SEASHELLS"
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"

POSITION_LIMITS = {
    AMETHYSTS: 20,
    STARFRUIT: 20,
}


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]]
            )

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class AmethystConfigs:
    def __init__(
        self,
        listing: Listing,
        price: int,
        mm_spread: int,
        quantity: int,
        threshold: int,
    ):
        self.listing = listing
        self.price = price
        self.mm_spread = mm_spread
        self.quantity = quantity
        self.threshold = threshold


class AmethystTrader:
    def __init__(self, configs: AmethystConfigs) -> None:
        self.product = configs.listing.product
        self.price = configs.price
        self.mm_spread = configs.mm_spread
        self.quantity = configs.quantity
        self.threshold = configs.threshold

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = []
        conversions = 0
        trader_data = ""

        # #mean and stdev of all past trades since last iteration (excluding our own)
        # market_prices = [] #trades not including our own
        # for trade in state.market_trades.values():
        #     if trade.Symbol == AMETHYSTS:
        #         market_prices.extend(trade.price)
        # mean = statistics.mean(market_prices)
        # std_dev = statistics.stdev(market_prices)

        # #mean and stdev of all past trades since last interation (including our own)
        # market_prices = [] #trades not including our own
        # for trade in state.market_trades.values():
        #     if trade.Symbol == AMETHYSTS:
        #         market_prices.extend(trade.price)
        # for trade in state.own_trades.values():
        #     if trade.Symbol == AMETHYSTS:
        #         market_prices.extend(trade.price)
        # mean = statistics.mean(market_prices)
        # std_dev = statistics.stdev(market_prices)

        # #weighted mean/stdev accounting for quantity including only market trades since last iteration
        # market_prices = [] #trades not including our own
        # for trade in state.market_trades.values():
        #     if trade.Symbol == AMETHYSTS:
        #         for i in range (trade.quantity):
        #             market_prices.extend(trade.price)
        # mean = statistics.mean(market_prices)
        # std_dev = statistics.stdev(market_prices)

        # #weighted mean/stdev accounting for quantity including all trades since last iteration
        # market_prices = [] #trades not including our own
        # for trade in state.market_trades.values():
        #     if trade.Symbol == AMETHYSTS:
        #         for i in range (trade.quantity):
        #             market_prices.extend(trade.price)
        # for trade in state.own_trades.values():
        #     if trade.Symbol == AMETHYSTS:
        #         for i in range (trade.quantity):
        #             market_prices.extend(trade.price)
        # mean = statistics.mean(market_prices)
        # std_dev = statistics.stdev(market_prices)

        # if any outstanding orders exist that are one standard deviation away in an advantageuous
        # direction, immediately take the trade
        outstanding_sells = state.order_depths.get(AMETHYSTS).sell_orders
        outstanding_buys = state.order_depths.get(AMETHYSTS).buy_orders

        # current weighted average of all outstanding market orders
        # market_prices = []
        # for order in outstanding_sells.items():
        #     for i in range (order[1]):
        #         market_prices.extend(order[0])
        # for order in outstanding_buys.items():
        #     for i in range (order[1]):
        #         market_prices.extend(order[0])
        # mean = statistics.mean(market_prices)
        # std_dev = statistics.stdev(market_prices)

        # check mean and standard deviation on outstanding orders to check if we want to buy/sell
        # buy
        # pos = state.position.get(self.product, 0)

        # for level in outstanding_buys.keys():
        #     quantity = outstanding_buys[level]
        #     if level > 10_000:
        #         logger.print(f"SELL {self.product}, ask_price={level}, ask_quantity{-quantity}")
        #         orders.append(Order(self.product, level, -quantity))
        #     if level == 10_000 and pos > 0:
        #         logger.print(f"SELL {self.product}, ask_price={level}, ask_quantity{-pos}")
        #         orders.append(Order(self.product, level, -max(quantity, pos)))

        # for level in outstanding_sells.keys():
        #     quantity = outstanding_sells[level]
        #     if level < 10_000:
        #         logger.print(f"BUY {self.product}, bid_price={level}, bid_quantity{quantity}")
        #         orders.append(Order(self.product, level, quantity))
        #     if level == 10_000 and pos < 0:
        #         logger.print(f"BUY {self.product}, bid_price={level}, bid_quantity{pos}")
        #         orders.append(Order(self.product, level, max(quantity, -pos)))

        # buy
        position_copy = state.position.get(self.product, 0)
        for key, value in outstanding_sells.items():
            value = -value
            if key == 10000:
                if position_copy < -self.threshold:
                    amount_bought = value
                    if -self.threshold - position_copy < value:
                        amount_bought = -self.threshold - position_copy
                    orders.append(Order(AMETHYSTS, key, amount_bought))
                    logger.print(
                        f"BUY {AMETHYSTS}, bid_price={key}, bid_quantity{amount_bought}"
                    )
                    position_copy += amount_bought
            if key < 10000:
                amount_bought = value
                if value + position_copy > POSITION_LIMITS[AMETHYSTS]:
                    amount_bought = POSITION_LIMITS[AMETHYSTS] - position_copy
                if amount_bought > 0:
                    orders.append(Order(AMETHYSTS, key, amount_bought))
                    logger.print(
                        f"BUY {AMETHYSTS}, bid_price={key}, bid_quantity{amount_bought}"
                    )
                position_copy += amount_bought

        # sell
        position_copy = state.position.get(self.product, 0)
        for key, value in outstanding_buys.items():
            if key == 10000:
                if position_copy > self.threshold:
                    amount_sold = value
                    if position_copy - self.threshold < value:
                        amount_sold = position_copy - self.threshold
                    orders.append(Order(AMETHYSTS, key, -amount_sold))
                    logger.print(
                        f"SELL {AMETHYSTS}, bid_price={key}, bid_quantity{-amount_sold}"
                    )
                    position_copy -= amount_sold
            if key > 10000:
                amount_sold = value
                if position_copy - value < -POSITION_LIMITS[AMETHYSTS]:
                    amount_sold = position_copy - POSITION_LIMITS[AMETHYSTS]
                if amount_sold > 0:
                    orders.append(Order(AMETHYSTS, key, -amount_sold))
                    logger.print(
                        f"SELL {AMETHYSTS}, bid_price={key}, bid_quantity{-amount_sold}"
                    )
                position_copy -= amount_sold
        return orders, conversions, trader_data


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # initialize configs
        amethyst_configs = AmethystConfigs(
            Listing(symbol=AMETHYSTS, product=AMETHYSTS, denomination=SEASHELLS),
            price=10_000,
            mm_spread=2,
            quantity=5,
            threshold=17,
        )

        # initialize traders
        amethyst_trader = AmethystTrader(amethyst_configs)

        # run traders
        (
            amethyst_orders,
            amethyst_conversions,
            amethyst_trader_data,
        ) = amethyst_trader.run(state)

        # create orders, conversions and trader_data
        orders = {}
        conversions = 0
        trader_data = ""

        orders[AMETHYSTS] = amethyst_orders
        conversions += amethyst_conversions
        trader_data += amethyst_trader_data

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
