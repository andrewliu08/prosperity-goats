import json
from collections import OrderedDict
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    Product,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import Any, Dict, Optional

import numpy as np


SEASHELLS = "SEASHELLS"
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
ORCHIDS = "ORCHIDS"

PRODUCTS = [AMETHYSTS, STARFRUIT, ORCHIDS]

POSITION_LIMITS = {
    AMETHYSTS: 20,
    STARFRUIT: 20,
    ORCHIDS: 100,
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
        self.conversions = 0
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

    def get_conv_observations(
        self,
    ) -> tuple[float, float, float, float, float, float, float]:
        """
        Returns the observation data.
        (bid_price, ask_price, transport_fees, export_tariff, import_tariff, sunlight, humidity)
        """
        conv_observations = self.state.observations.conversionObservations[self.product]
        return (
            conv_observations.bidPrice,
            conv_observations.askPrice,
            conv_observations.transportFees,
            conv_observations.exportTariff,
            conv_observations.importTariff,
            conv_observations.sunlight,
            conv_observations.humidity,
        )
    
    def set_conversion(self, conversion: int) -> None:
        """
        Set a conversion value.
        Conversion can't be zero.
        """
        position = self.get_position()
        if position < 0:
            assert 1 <= conversion and conversion <= -position, f"Invalid conversion value: {conversion=}, {position=}"
        elif position > 0:
            assert -position <= conversion and conversion <= -1, f"Invalid conversion value: {conversion=}, {position=}"
        else:
            assert False, f"ERROR: {position=}, cannot do conversion"

        self.conversions = conversion


logger = Logger()


class AmethystConfigs:
    def __init__(
        self,
        listing: Listing,
        manager: Manager,
        price: int,
    ):
        self.listing = listing
        self.manager = manager
        self.price = price


class AmethystTrader:
    def __init__(self, configs: AmethystConfigs) -> None:
        self.product = configs.listing.product
        self.manager = configs.manager
        self.price = configs.price

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        position = self.manager.get_position()

        # Buy Orders
        sell_orders = self.manager.get_sell_orders()
        exp_pos = position
        for price, qty in sell_orders.items():
            if price < self.price or (position < 0 and price == self.price):
                buy_amount = min(self.manager.max_buy_amount(exp_pos), -qty)
                if buy_amount > 0:
                    self.manager.place_buy_order(price, buy_amount)
                    exp_pos += buy_amount

        max_buy_amount = self.manager.max_buy_amount(exp_pos)
        best_buy_order = self.manager.get_best_buy_order()
        best_buy_price = best_buy_order[0] if best_buy_order is not None else self.price
        if max_buy_amount > 0:
            if position < 0:
                price = min(best_buy_price + 2, self.price - 1)
                self.manager.place_buy_order(price, max_buy_amount)
                exp_pos += max_buy_amount
            elif position > 15:
                price = min(best_buy_price, self.price - 1)
                self.manager.place_buy_order(price, max_buy_amount)
                exp_pos += max_buy_amount
            else:
                price = min(best_buy_price + 1, self.price - 1)
                self.manager.place_buy_order(price, max_buy_amount)
                exp_pos += max_buy_amount

        # Sell Orders
        buy_orders = self.manager.get_buy_orders()
        exp_pos = position
        for price, qty in buy_orders.items():
            if price > self.price or (position > 0 and price == self.price):
                sell_amount = max(self.manager.max_sell_amount(exp_pos), qty)
                if sell_amount < 0:
                    self.manager.place_sell_order(price, sell_amount)
                    exp_pos += sell_amount

        max_sell_amount = self.manager.max_sell_amount(exp_pos)
        best_sell_order = self.manager.get_best_sell_order()
        best_sell_price = (
            best_sell_order[0] if best_sell_order is not None else self.price
        )
        if max_sell_amount < 0:
            if position > 0:
                price = max(best_sell_price - 2, self.price + 1)
                self.manager.place_sell_order(price, max_sell_amount)
                exp_pos += max_sell_amount
            elif position < -15:
                price = max(best_sell_price, self.price + 1)
                self.manager.place_sell_order(price, max_sell_amount)
                exp_pos += max_sell_amount
            else:
                price = max(best_sell_price - 1, self.price + 1)
                self.manager.place_sell_order(price, max_sell_amount)
                exp_pos += max_sell_amount


class StarfruitConfigs:
    def __init__(
        self,
        listing: Listing,
        manager: Manager,
    ):
        self.listing = listing
        self.manager = manager


class StarfruitTrader:
    def __init__(self, configs: StarfruitConfigs) -> None:
        self.product = configs.listing.product
        self.manager = configs.manager

    def run(self, state: TradingState) -> None:
        future_price = self.manager.get_VWAP()
        position = self.manager.get_position()

        # Buy Orders
        exp_pos = self.manager.get_position()
        sell_orders = self.manager.get_sell_orders()
        for price, qty in sell_orders.items():
            if price < future_price or (position < 0 and price == future_price):
                buy_amount = min(self.manager.max_buy_amount(exp_pos), -qty)
                if buy_amount > 0:
                    self.manager.place_buy_order(price, buy_amount)
                    exp_pos += buy_amount

        price, qty = self.manager.get_best_buy_order()
        price = min(price + 1, future_price - 1)
        buy_amount = self.manager.max_buy_amount(exp_pos)
        if buy_amount > 0:
            self.manager.place_buy_order(price, buy_amount)

        # Sell Orders
        exp_pos = self.manager.get_position()
        buy_orders = self.manager.get_buy_orders()
        for price, qty in buy_orders.items():
            if price > future_price or (position > 0 and price == future_price):
                sell_amount = max(self.manager.max_sell_amount(exp_pos), -qty)
                if sell_amount < 0:
                    self.manager.place_sell_order(price, sell_amount)
                    exp_pos += sell_amount

        price, qty = self.manager.get_best_sell_order()
        price = max(price - 1, future_price + 1)
        sell_amount = self.manager.max_sell_amount(exp_pos)
        if sell_amount < 0:
            self.manager.place_sell_order(price, sell_amount)


class OrchidConfigs:
    def __init__(
        self,
        listing: Listing,
        manager: Manager,
    ):
        self.listing = listing
        self.manager = manager


class OrchidTrader:
    def __init__(self, configs: OrchidConfigs) -> None:
        self.product = configs.listing.product
        self.manager = configs.manager

    def run(self, state: TradingState) -> None:
        (
            bid_price,
            ask_price,
            transport_fees,
            export_tariff,
            import_tariff,
            sunlight,
            humidity,
        ) = self.manager.get_conv_observations()
        conv_bid_price = bid_price - export_tariff - transport_fees
        conv_ask_price = ask_price + import_tariff + transport_fees
        position = self.manager.get_position()


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # initialize managers
        managers = {product: Manager(product, state) for product in PRODUCTS}

        # initialize configs
        amethyst_configs = AmethystConfigs(
            Listing(symbol=AMETHYSTS, product=AMETHYSTS, denomination=SEASHELLS),
            manager=managers[AMETHYSTS],
            price=10_000,
        )
        starfruit_configs = StarfruitConfigs(
            Listing(symbol=STARFRUIT, product=STARFRUIT, denomination=SEASHELLS),
            manager=managers[STARFRUIT],
        )
        orchid_configs = OrchidConfigs(
            Listing(symbol=STARFRUIT, product=STARFRUIT, denomination=SEASHELLS),
            manager=managers[ORCHIDS],
        )

        # initialize traders
        amethyst_trader = AmethystTrader(amethyst_configs)
        starfruit_trader = StarfruitTrader(starfruit_configs)
        orchid_trader = OrchidTrader(orchid_configs)

        # run traders
        amethyst_trader.run(state)
        starfruit_trader.run(state)
        orchid_trader.run(state)

        # create orders, conversions and trader_data
        orders = {}
        conversions = 0
        new_trader_data = {}

        orders[AMETHYSTS] = amethyst_trader.manager.pending_orders()
        orders[STARFRUIT] = starfruit_trader.manager.pending_orders()
        orders[ORCHIDS] = orchid_trader.manager.pending_orders()

        conversions = managers[ORCHIDS].conversions

        for product in PRODUCTS:
            new_trader_data.update(managers[product].get_new_trader_data())
        new_trader_data = json.dumps(new_trader_data)

        logger.flush(state, orders, conversions, new_trader_data)
        return orders, conversions, new_trader_data
