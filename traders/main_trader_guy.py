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
from typing import Any
from collections import OrderedDict

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
    def __init__(self, product, state):
        self.product = product
        self.state = state
        self.orders = []

    def get_position(self):
        return self.state.position.get(self.product, 0)
    
    def get_buy_orders(self):
        """
        Returns an ordered and sorted (from best to worst) dictionary of buy orders for the product.
        """
        return OrderedDict(sorted(self.state.order_depths[self.product].buy_orders.items(), reverse=True))
    
    def get_sell_orders(self):
        """
        Returns an ordered and sorted (from best to worst) dictionary of sell orders for the product.
        """
        return OrderedDict(sorted(self.state.order_depths[self.product].sell_orders.items()))
    
    def get_best_buy_order(self):
        """
        Returns the (price, quantity) for the best buy order for the product.
        """
        buy_orders = self.get_buy_orders()
        assert len(buy_orders) > 0, "no buy orders found"

        return list(buy_orders.items())[0]
    
    def get_best_sell_order(self):
        """
        Returns the (price, quantity) for the best sell order for the product.
        """
        sell_orders = self.get_sell_orders()
        assert len(sell_orders) > 0, "no sell orders found"

        return list(sell_orders.items())[0]
    
    def place_order(self, price, quantity):
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
    
    def place_buy_order(self, price, quantity):
        assert quantity > 0, f"buy order quantity must be positive. {quantity=}"
        assert quantity <= self.max_buy_amount(), f"buy order quantity exceeds position limit. {quantity=}, {self.max_buy_amount()=}"

        return self.place_order(price, quantity)
    
    def place_sell_order(self, price, quantity):
        assert quantity < 0, f"sell order quantity must be negative. {quantity=}"
        assert quantity >= self.max_sell_amount(), f"sell order quantity exceeds position limit. {quantity=}, {self.max_sell_amount()=}"

        return self.place_order(price, quantity)
    
    def pending_orders(self):
        ret = [order for order in self.orders if order.quantity != 0]
        self.orders = []
        return ret

    def max_buy_amount(self, position=None):
        """
        Returns the maximum quantity you can buy.
        position: The position you want to calculate the maximum buy amount for. If None, the current position is used.
        """
        if position is None:
            position = self.get_position()
        return POSITION_LIMITS[self.product] - position
    
    def max_sell_amount(self, position=None):
        """
        Returns the minimum quantity you can sell (since it is a negative number).
        position: The position you want to calculate the minimum sell amount for. If None, the current position is used.
        """
        if position is None:
            position = self.get_position()
        return -POSITION_LIMITS[self.product] - position
    
    def get_mid_price(self):
        """
        Returns (best_buy_price + best_sell_price) / 2 rounded to the nearest int.
        """
        best_buy_order = self.get_best_buy_order()
        best_sell_order = self.get_best_sell_order()
        return round((best_buy_order[0] + best_sell_order[0]) / 2.0)
    
    def get_VWAP(self):
        """
        Returns the VWAP (weighted average of price) rounded to the nearest int.
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
        return round(total / volume)

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

    def run(self, state: TradingState) -> None:
        orders = []
        conversions = 0
        trader_data = ""

        bid_price = self.price - self.mm_spread // 2
        bid_quantity = min(self.quantity, self.manager.max_buy_amount())
        if bid_quantity != 0:
            self.manager.place_buy_order(bid_price, bid_quantity)

        ask_price = self.price + self.mm_spread // 2
        ask_quantity = max(-self.quantity, self.manager.max_sell_amount())
        if ask_quantity != 0:
            self.manager.place_sell_order(ask_price, ask_quantity)


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
            manager=managers[AMETHYSTS]
        )

        # initialize traders
        amethyst_trader = AmethystTrader(amethyst_configs)

        # run traders
        amethyst_trader.run(state)

        # create orders, conversions and trader_data
        orders = {}
        conversions = 0
        trader_data = ""

        orders[AMETHYSTS] = amethyst_trader.manager.pending_orders()

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
    