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
from typing import Any, Optional

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
    ):
        self.listing = listing
        self.price = price
        self.mm_spread = mm_spread
        self.quantity = quantity


class AmethystTrader:
    def __init__(self, configs: AmethystConfigs) -> None:
        self.product = configs.listing.product
        self.price = configs.price
        self.mm_spread = configs.mm_spread
        self.quantity = configs.quantity

    def calc_reservation_price(self, position: int) -> int:
        reservation_price = self.price - int(position * 0.1)
        return reservation_price

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = []
        conversions = 0
        trader_data = ""

        position = state.position.get(self.product, 0)
        reservation_price = self.calc_reservation_price(
            state.position.get(self.product, 0)
        )

        bid_price = reservation_price - self.mm_spread // 2
        bid_quantity = min(self.quantity, POSITION_LIMITS[AMETHYSTS] - position)
        if bid_quantity > 0:
            logger.print(f"BUY {self.product}, {bid_price=}, {bid_quantity=}")
            orders.append(Order(self.product, bid_price, bid_quantity))

        ask_price = reservation_price + self.mm_spread // 2
        ask_quantity = max(-self.quantity, -POSITION_LIMITS[AMETHYSTS] - position)
        if ask_quantity < 0:
            logger.print(f"SELL {self.product}, {ask_price=}, {ask_quantity=}")
            orders.append(Order(self.product, ask_price, ask_quantity))

        return orders, conversions, trader_data

    # def run1(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
    #     orders = []
    #     conversions = 0
    #     trader_data = ""

    #     position = state.position.get(self.product, 0)

    #     bid_quantity = min(self.quantity, POSITION_LIMITS[AMETHYSTS] - position)
    #     ask_quantity = max(-self.quantity, -POSITION_LIMITS[AMETHYSTS] - position)

    #     bid_prices = sorted(state.order_depths[AMETHYSTS].buy_orders.keys())
    #     if len(bid_prices) == 0:
    #         bid_price = self.price - self.mm_spread // 2
    #     else:
    #         bid_price = bid_prices[-1] + 2

    #     ask_prices = sorted(state.order_depths[AMETHYSTS].sell_orders.keys())
    #     if len(ask_prices) == 0:
    #         ask_price = self.price + self.mm_spread // 2
    #     else:
    #         ask_price = ask_prices[0] - 1

    #     if bid_quantity > 0:
    #         logger.print(f"BUY {self.product}, {bid_price=}, {bid_quantity=}")
    #         orders.append(Order(self.product, bid_price, bid_quantity))
    #     if ask_quantity < 0:
    #         logger.print(f"SELL {self.product}, {ask_price=}, {ask_quantity=}")
    #         orders.append(Order(self.product, ask_price, ask_quantity))

    #     return orders, conversions, trader_data


class StarfruitConfigs:
    def __init__(
        self,
        listing: Listing,
        mm_spread: int,
        quantity: int,
    ):
        self.listing = listing
        self.mm_spread = mm_spread
        self.quantity = quantity


class StarfruitTrader:
    def __init__(self, configs: StarfruitConfigs) -> None:
        self.product = configs.listing.product
        self.mm_spread = configs.mm_spread
        self.quantity = configs.quantity

    def calc_reservation_price(self, price: int, position: int) -> int:
        reservation_price = price - int(position * 0.09)
        return reservation_price
    
    def vwap(self, state: TradingState) -> Optional[float]:
        order_depths = state.order_depths.get(self.product)
        if order_depths is None:
            return None
        
        buy_orders = order_depths.buy_orders
        sell_orders = order_depths.sell_orders
        total, volume = 0, 0
        for price, qty in buy_orders.items():
            total += price * qty
            volume += qty
        for price, qty in sell_orders.items():
            total += price * -qty
            volume += -qty

        if volume == 0:
            return None
        return total / volume
    
    def highest_bid(self, state: TradingState) -> Optional[int]:
        buy_orders = state.order_depths[self.product].buy_orders
        return None if len(buy_orders) == 0 else max(buy_orders.keys())

    def lowest_ask(self, state: TradingState) -> Optional[int]:
        sell_orders = state.order_depths[self.product].sell_orders
        return None if len(sell_orders) == 0 else min(sell_orders.keys())
    
    def mid_price(self, state: TradingState) -> Optional[float]:
        best_bid = self.highest_bid(state)
        best_ask = self.lowest_ask(state)
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = []
        conversions = 0
        trader_data = ""

        position = state.position.get(self.product, 0)
        mid_price = self.vwap(state)
        if mid_price is None:
            return orders, conversions, trader_data
        reservation_price = self.calc_reservation_price(
            int(mid_price),
            state.position.get(self.product, 0),
        )

        bid_price = reservation_price - self.mm_spread // 2
        bid_quantity = min(self.quantity, POSITION_LIMITS[STARFRUIT] - position)
        if bid_quantity > 0:
            logger.print(f"BUY {self.product}, {bid_price=}, {bid_quantity=}")
            orders.append(Order(self.product, bid_price, bid_quantity))

        ask_price = reservation_price + self.mm_spread // 2
        ask_quantity = max(-self.quantity, -POSITION_LIMITS[STARFRUIT] - position)
        if ask_quantity < 0:
            logger.print(f"SELL {self.product}, {ask_price=}, {ask_quantity=}")
            orders.append(Order(self.product, ask_price, ask_quantity))

        return orders, conversions, trader_data


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # initialize configs
        amethyst_configs = AmethystConfigs(
            Listing(symbol=AMETHYSTS, product=AMETHYSTS, denomination=SEASHELLS),
            price=10_000,
            mm_spread=2,
            quantity=40,
        )
        starfruit_configs = StarfruitConfigs(
            Listing(symbol=STARFRUIT, product=STARFRUIT, denomination=SEASHELLS),
            mm_spread=4,
            quantity=30,
        )

        # initialize traders
        amethyst_trader = AmethystTrader(amethyst_configs)
        starfruit_trader = StarfruitTrader(starfruit_configs)

        # run traders
        # (
        #     amethyst_orders,
        #     amethyst_conversions,
        #     amethyst_trader_data,
        # ) = amethyst_trader.run(state)
        (
            starfruit_orders,
            starfruit_conversions,
            starfruit_trader_data,
        ) = starfruit_trader.run(state)

        # create orders, conversions and trader_data
        orders = {}
        conversions = 0
        trader_data = ""

        # orders[AMETHYSTS] = amethyst_orders
        # conversions += amethyst_conversions
        # trader_data += amethyst_trader_data
        orders[STARFRUIT] = starfruit_orders
        conversions += starfruit_conversions
        trader_data += starfruit_trader_data

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
