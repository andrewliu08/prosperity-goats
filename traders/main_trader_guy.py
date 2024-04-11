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
from typing import Any, Dict, Optional
from collections import OrderedDict

import numpy as np

Product = str

SEASHELLS = "SEASHELLS"
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"

PRODUCTS = [AMETHYSTS, STARFRUIT]

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


class Manager:
    def __init__(self, product: Product, state: TradingState) -> None:
        self.product = product
        self.state = state
        self.orders = []
        self.trader_data: Dict[str, Any] = (
            json.loads(self.state.traderData) if self.state.traderData else {}
        )
        self.new_trader_data: Dict[str, Any] = {}

    def get_position(self) -> int:
        return self.state.position.get(self.product, 0)

    def get_buy_orders(self) -> OrderedDict[int, int]:
        """
        Returns the (price, quantity) of buy orders for the product.
        Returns an OrderedDict that's sorted based on price (from best to worst).
        """
        return OrderedDict(
            sorted(
                self.state.order_depths[self.product].buy_orders.items(), reverse=True
            )
        )

    def get_sell_orders(self) -> OrderedDict[int, int]:
        """
        Returns the (price, quantity) of sell orders for the product.
        Returns an OrderedDict that's sorted based on price (from best to worst).
        """
        return OrderedDict(
            sorted(self.state.order_depths[self.product].sell_orders.items())
        )

    def get_best_buy_order(self) -> Optional[tuple[int, int]]:
        """
        Returns the price, quantity for the best buy order for the product.
        """
        buy_orders = self.get_buy_orders()
        if len(buy_orders) == 0:
            return None

        return list(buy_orders.items())[0]

    def get_best_sell_order(self) -> Optional[tuple[int, int]]:
        """
        Returns the (price, quantity) for the best sell order for the product.
        """
        sell_orders = self.get_sell_orders()
        if len(sell_orders) == 0:
            return None

        return list(sell_orders.items())[0]

    def place_order(self, price: int, quantity: int) -> None:
        """
        DO NOT USE. Use place_buy_order or place_sell_order instead.
        Place an order for the product with the given price and quantity.
        """
        assert quantity != 0, "cannot place an order with quantity 0"

        if quantity > 0:
            logger.print(f"BUY {self.product}, {price=}, {quantity=}")
        else:
            logger.print(f"SELL {self.product}, {price=}, {quantity=}")
        self.orders.append(Order(self.product, price, quantity))

    def place_buy_order(self, price: int, quantity: int) -> None:
        assert quantity > 0, f"buy order quantity must be positive. {quantity=}"
        assert (
            quantity <= self.max_buy_amount()
        ), f"buy order quantity exceeds position limit. {quantity=}, {self.max_buy_amount()=}"

        self.place_order(price, quantity)

    def place_sell_order(self, price: int, quantity: int) -> None:
        assert quantity < 0, f"sell order quantity must be negative. {quantity=}"
        assert (
            quantity >= self.max_sell_amount()
        ), f"sell order quantity exceeds position limit. {quantity=}, {self.max_sell_amount()=}"

        self.place_order(price, quantity)

    def pending_orders(self) -> list[Order]:
        ret = [order for order in self.orders if order.quantity != 0]
        self.orders = []
        return ret

    def max_buy_amount(self, position: Optional[int] = None) -> int:
        """
        Returns the maximum quantity you can buy.
        position: The position you want to calculate the maximum buy amount for. If None, the current position is used.
        """
        if position is None:
            position = self.get_position()
        return POSITION_LIMITS[self.product] - position

    def max_sell_amount(self, position: Optional[int] = None) -> int:
        """
        Returns the minimum quantity you can sell (since it is a negative number).
        position: The position you want to calculate the minimum sell amount for. If None, the current position is used.
        """
        if position is None:
            position = self.get_position()
        return -POSITION_LIMITS[self.product] - position

    def get_mid_price(self) -> Optional[int]:
        """
        Returns (best_buy_price + best_sell_price) / 2 rounded to the nearest int.
        Returns None if there are no buy or sell orders.
        """
        best_buy_order = self.get_best_buy_order()
        best_sell_order = self.get_best_sell_order()
        if best_buy_order is None and best_sell_order is None:
            return None
        if best_buy_order is None:
            return best_sell_order[0]
        if best_sell_order is None:
            return best_buy_order[0]
        return round((best_buy_order[0] + best_sell_order[0]) / 2.0)

    def get_VWAP(self) -> Optional[int]:
        """
        Returns the VWAP (weighted average of price) rounded to the nearest int.
        Returns None if there are no buy or sell orders.
        """
        buy_orders = self.get_buy_orders()
        sell_orders = self.get_sell_orders()
        total, volume = 0, 0
        for price, qty in buy_orders.items():
            total += price * qty
            volume += qty
        for price, qty in sell_orders.items():
            total += price * -qty
            volume += -qty

        if volume == 0:
            return None
        return round(total / volume)

    def add_trader_data(self, key: str, value: Any) -> None:
        self.new_trader_data[key] = value

    def get_new_trader_data(self) -> Dict[str, Any]:
        """
        Used to update trader_data for the next iteration.
        """
        return self.new_trader_data


logger = Logger()


class AmethystConfigs:
    def __init__(
        self,
        listing: Listing,
        price: int,
        mm_spread: int,
        quantity: int,
        manager: Manager,
    ):
        self.listing = listing
        self.price = price
        self.mm_spread = mm_spread
        self.quantity = quantity
        self.manager = manager


class AmethystTrader:
    def __init__(self, configs: AmethystConfigs) -> None:
        self.product = configs.listing.product
        self.price = configs.price
        self.mm_spread = configs.mm_spread
        self.quantity = configs.quantity
        self.manager = configs.manager

    def position_adjustment(self, adjustments: list[int], position: int):
        lim = POSITION_LIMITS[self.product]
        cutoffs = np.linspace(-lim, lim, len(adjustments) + 1)
        for adj, cutoff in zip(adjustments, cutoffs[1:-1]):
            if position <= cutoff:
                return adj
        return adjustments[-1]

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        buy_quota = self.manager.max_buy_amount()
        sell_quota = self.manager.max_sell_amount()

        buy_book = [12, 8]
        spread = 1
        # spread of 0 seems to be better for day -2, 1 better for day -1, and day 0
        adj = self.position_adjustment(
            [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            self.manager.get_position(),
        )

        bp = 10000 - spread - adj
        sp = 10000 + spread - adj

        for i, qty in enumerate(buy_book):
            q = min(qty, buy_quota)
            if q != 0:
                self.manager.place_buy_order(bp - i, q)
            buy_quota -= q
        if buy_quota > 0:
            self.manager.place_buy_order(bp - len(buy_book), buy_quota)

        for i, qty in enumerate(buy_book):
            q = max(-qty, sell_quota)
            if q != 0:
                self.manager.place_sell_order(sp + i, q)
            sell_quota -= q
        if sell_quota < 0:
            self.manager.place_sell_order(sp + len(buy_book), sell_quota)


class StarfruitConfigs:
    def __init__(
        self,
        listing: Listing,
        manager: Manager,
        coefs: list[float],
        intercept: float,
        mm_spread: int,
        inventory_adjustment: float,
    ):
        self.listing = listing
        self.manager = manager

        # Taker:
        self.coefs = coefs
        self.intercept = intercept

        # Maker:
        self.mm_spread = mm_spread
        self.inventory_adjustment = inventory_adjustment


class StarfruitTrader:
    def __init__(self, configs: StarfruitConfigs) -> None:
        self.product = configs.listing.product
        self.manager = configs.manager
        self.coefs = configs.coefs
        self.intercept = configs.intercept
        self.mm_spread = configs.mm_spread
        self.inventory_adjustment = configs.inventory_adjustment

    def calc_reservation_price(self, price: int, position: int) -> int:
        reservation_price = price - int(position * self.inventory_adjustment)
        return reservation_price

    def run(self, state: TradingState) -> None:
        SF_PREV_PRICES_DIM = 4
        orders = []

        # Pre-Processing
        trader_data = self.manager.trader_data
        mid_price = self.manager.get_VWAP()

        sf_prev_prices = trader_data.get("sf_prev_prices", [])  # stores the VWAP
        if len(sf_prev_prices) == SF_PREV_PRICES_DIM:
            sf_prev_prices = sf_prev_prices[1:]
        sf_prev_prices.append(mid_price)

        # Linear Regression
        if len(sf_prev_prices) == SF_PREV_PRICES_DIM:
            X = sf_prev_prices  # lag_x
            future_price = self.intercept
            for i in range(len(X)):
                future_price += X[i] * self.coefs[i]
        else:
            X = []

        # Buy Orders
        max_buy_amount = self.manager.max_buy_amount()
        total_buy_amount = 0

        if X != []:
            sell_orders = self.manager.get_sell_orders()
            for price, quantity in sell_orders.items():
                if total_buy_amount >= max_buy_amount:
                    break

                if price < future_price:
                    buy_amount = min(max_buy_amount - total_buy_amount, -quantity)
                    if buy_amount > 0:
                        self.manager.place_buy_order(price, buy_amount)
                        total_buy_amount += buy_amount

        reservation_price = self.calc_reservation_price(
            int(future_price if X != [] else mid_price), self.manager.get_position()
        )
        bid_price = reservation_price - self.mm_spread // 2
        bid_quantity = max_buy_amount - total_buy_amount
        if bid_quantity > 0:
            self.manager.place_buy_order(bid_price, bid_quantity)

        # Sell Orders
        max_sell_amount = self.manager.max_sell_amount()
        total_sell_amount = 0

        if X != []:
            buy_orders = self.manager.get_buy_orders()
            for price, quantity in buy_orders.items():
                if total_sell_amount <= max_sell_amount:
                    break

                if price > future_price:
                    sell_amount = min(max_sell_amount - total_sell_amount, quantity)
                    if sell_amount < 0:
                        self.manager.place_sell_order(price, sell_amount)
                        total_sell_amount += sell_amount

        ask_price = bid_price + self.mm_spread
        ask_quantity = max_sell_amount - total_sell_amount
        if ask_quantity < 0:
            self.manager.place_sell_order(ask_price, ask_quantity)

        # Update trader data
        self.manager.add_trader_data("sf_prev_prices", sf_prev_prices)


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # initialize managers
        managers = {product: Manager(product, state) for product in PRODUCTS}

        # initialize configs
        amethyst_configs = AmethystConfigs(
            Listing(symbol=AMETHYSTS, product=AMETHYSTS, denomination=SEASHELLS),
            price=10_000,
            mm_spread=2,
            quantity=5,
            manager=managers[AMETHYSTS],
        )
        starfruit_configs = StarfruitConfigs(
            Listing(symbol=STARFRUIT, product=STARFRUIT, denomination=SEASHELLS),
            manager=managers[STARFRUIT],
            coefs=[
                0.030847739217679665,
                0.047536040306744326,
                0.22183513277646527,
                0.6996451923517073,
            ],  # [lag3, lag2, lag1, lag0] VWAP ALL 100%
            intercept=0.6872903013800169,
            mm_spread=4,
            inventory_adjustment=0.09,
        )

        # initialize traders
        amethyst_trader = AmethystTrader(amethyst_configs)
        starfruit_trader = StarfruitTrader(starfruit_configs)

        # run traders
        amethyst_trader.run(state)
        starfruit_trader.run(state)

        # create orders, conversions and trader_data
        orders = {}
        conversions = 0
        new_trader_data = {}

        orders[AMETHYSTS] = amethyst_trader.manager.pending_orders()
        orders[STARFRUIT] = starfruit_trader.manager.pending_orders()

        for product in PRODUCTS:
            new_trader_data.update(managers[product].get_new_trader_data())
        new_trader_data = json.dumps(new_trader_data)

        logger.flush(state, orders, conversions, new_trader_data)
        return orders, conversions, new_trader_data
