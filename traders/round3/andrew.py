import json
import jsonpickle
import math
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
CHOCOLATE = "CHOCOLATE"
STRAWBERRIES = "STRAWBERRIES"
ROSES = "ROSES"
GIFT_BASKET = "GIFT_BASKET"

PRODUCTS = [AMETHYSTS, STARFRUIT, ORCHIDS, CHOCOLATE, STRAWBERRIES, ROSES, GIFT_BASKET]

POSITION_LIMITS = {
    AMETHYSTS: 20,
    STARFRUIT: 20,
    ORCHIDS: 100,
    CHOCOLATE: 250,
    STRAWBERRIES: 350,
    ROSES: 60,
    GIFT_BASKET: 60,
}

INVENTORY_COST = 0.1

BASKET_COMPOSITION = {CHOCOLATE: 4, STRAWBERRIES: 6, ROSES: 1}


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
        self.position = self.state.position.get(self.product, 0)
        self.trader_data: Dict[str, Any] = (
            jsonpickle.decode(self.state.traderData) if self.state.traderData else {}
        )
        self.new_trader_data: Dict[str, Any] = {}

    def get_position(self) -> int:
        return self.position

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
    
    def get_worst_buy_order(self) -> Optional[tuple[int, int]]:
        """
        Returns the price, quantity for the worst buy order for the product.
        """
        buy_orders = self.get_buy_orders()
        if len(buy_orders) == 0:
            return None

        return list(buy_orders.items())[-1]
    
    def get_worst_sell_order(self) -> Optional[tuple[int, int]]:
        """
        Returns the (price, quantity) for the worst sell order for the product.
        """
        sell_orders = self.get_sell_orders()
        if len(sell_orders) == 0:
            return None

        return list(sell_orders.items())[-1]

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
            assert (
                1 <= conversion and conversion <= -position
            ), f"Invalid conversion value: {conversion=}, {position=}"
        elif position > 0:
            assert (
                -position <= conversion and conversion <= -1
            ), f"Invalid conversion value: {conversion=}, {position=}"
        else:
            assert False, f"ERROR: {position=}, cannot do conversion"

        self.conversions = conversion
        self.position += conversion


logger = Logger()


class AmethystConfigs:
    def __init__(self, listing: Listing, manager: Manager, price: int):
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
    def __init__(self, listing: Listing, manager: Manager):
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
    def __init__(self, listing: Listing, arb_margin: float, manager: Manager):
        self.listing = listing
        self.arb_margin = arb_margin
        self.manager = manager


class OrchidTrader:
    def __init__(self, configs: OrchidConfigs) -> None:
        self.product = configs.listing.product
        self.arb_margin = configs.arb_margin
        self.manager = configs.manager

    def calc_inventory_cost(self, quantity: int, t: int) -> float:
        # Short positions don't incur inventory costs
        if quantity < 0:
            return 0
        return INVENTORY_COST * t

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

        # Arbitrage
        if position != 0:
            self.manager.set_conversion(-position)

        # Taker orders
        bid_pos = self.manager.get_position()
        sell_orders = self.manager.get_sell_orders()
        for price, qty in sell_orders.items():
            bid_quantity = min(self.manager.max_buy_amount(bid_pos), -qty)
            inventory_cost = self.calc_inventory_cost(bid_quantity, t=1)
            # buy at price now, sell at conv_bid_price next time_step
            if (
                conv_bid_price - price >= self.arb_margin + inventory_cost
                and bid_quantity > 0
            ):
                self.manager.place_buy_order(price, bid_quantity)
                bid_pos += bid_quantity

        # maker order expecting that conv_bid_price won't change much
        maker_buy_amount = self.manager.max_buy_amount(bid_pos)
        if maker_buy_amount > 0:
            inventory_cost = self.calc_inventory_cost(maker_buy_amount, t=1)
            self.manager.place_buy_order(
                math.floor(conv_bid_price - self.arb_margin - inventory_cost),
                maker_buy_amount,
            )

        ask_pos = self.manager.get_position()
        buy_orders = self.manager.get_buy_orders()
        for price, qty in buy_orders.items():
            ask_quantity = max(self.manager.max_sell_amount(ask_pos), -qty)
            inventory_cost = self.calc_inventory_cost(ask_quantity, t=1)
            # sell at price now, buy at conv_ask_price next time_step
            if (
                price - conv_ask_price >= self.arb_margin + inventory_cost
                and ask_quantity < 0
            ):
                self.manager.place_sell_order(price, ask_quantity)
                ask_pos += ask_quantity

        # maker order expecting that conv_ask_price won't change much
        maker_sell_amount = self.manager.max_sell_amount(ask_pos)
        if maker_sell_amount < 0:
            inventory_cost = self.calc_inventory_cost(maker_sell_amount, t=1)
            self.manager.place_sell_order(
                math.ceil(conv_ask_price + self.arb_margin + inventory_cost),
                maker_sell_amount,
            )


class BasketPairConfigs:
    def __init__(
        self,
        managers: dict[Product, Manager],
        mean_diff: float,
        trade_signal: float,
        max_basket_position: int,
        basket_order_quantity: int,
        spreads: dict[Product, int],
    ):
        self.managers = managers
        self.mean_diff = mean_diff
        self.trade_signal = trade_signal
        self.max_basket_position = max_basket_position
        self.basket_order_quantity = basket_order_quantity
        self.spreads = spreads


class BasketPairTrader:
    def __init__(self, configs: OrchidConfigs) -> None:
        self.managers = configs.managers
        self.mean_diff = configs.mean_diff
        self.trade_signal = configs.trade_signal
        self.max_basket_position = configs.max_basket_position
        self.basket_order_quantity = configs.basket_order_quantity
        self.spreads = configs.spreads

    def summed_basket_price(self, prices: dict[Product, float]) -> Optional[float]:
        if any(price is None for price in prices.values()):
            return None
        return sum(
            prices[product] * BASKET_COMPOSITION[product]
            for product in BASKET_COMPOSITION
        )

    def basket_diff(self, prices: dict[Product, float]) -> Optional[float]:
        # summed_price = self.summed_basket_price(prices)
        # if summed_price is None or prices[GIFT_BASKET] is None:
        #     return None
        summed_price = prices[CHOCOLATE] * 3.84280387 + prices[STRAWBERRIES] * 6.17207762 + prices[ROSES] * 1.05295733 + 162.57251083700976
        return prices[GIFT_BASKET] - summed_price

    def calc_order_quantities(
        self, positions: dict[Product, int], long_pair: bool
    ) -> dict[Product, int]:
        quantities = {}
        if long_pair:
            # long the basket
            quantities[GIFT_BASKET] = min(
                self.basket_order_quantity,
                self.max_basket_position - positions[GIFT_BASKET],
            )
            # short the components
            for product in BASKET_COMPOSITION:
                # if product == "CHOCOLATE":
                quantities[product] = max(
                    -self.basket_order_quantity * BASKET_COMPOSITION[product],
                    -self.max_basket_position * BASKET_COMPOSITION[product]
                    - positions[product],
                    self.managers[product].max_sell_amount(),
                )
        else:
            # short the basket
            quantities[GIFT_BASKET] = max(
                -self.basket_order_quantity,
                -self.max_basket_position - positions[GIFT_BASKET],
            )
            # long the components
            for product in BASKET_COMPOSITION:
                quantities[product] = min(
                    self.basket_order_quantity * BASKET_COMPOSITION[product],
                    self.max_basket_position * BASKET_COMPOSITION[product]
                    - positions[product],
                    self.managers[product].max_buy_amount(),
                )

        return quantities

    def run_basket(self, state: TradingState) -> None:
        prices = {
            product: manager.get_VWAP() for product, manager in self.managers.items()
        }
        positions = {
            product: manager.get_position()
            for product, manager in self.managers.items()
        }

        price_diff = self.basket_diff(prices)
        if price_diff is None:
            return

        # Price diff is high, short the pair by shorting the basket
        # and longing the components
        # if price_diff > self.mean_diff + self.trade_signal:
        std = 75.81995910676206
        if price_diff > std * 0.45:
            quantities = self.calc_order_quantities(positions, long_pair=False)
            if quantities[GIFT_BASKET] < 0:
                self.managers[GIFT_BASKET].place_sell_order(
                    # math.ceil(prices[GIFT_BASKET] + self.spreads[GIFT_BASKET]),
                    self.managers[GIFT_BASKET].get_worst_buy_order()[0],
                    quantities[GIFT_BASKET],
                )
            # for product in BASKET_COMPOSITION:
            #     if quantities[product] > 0:
            #         self.managers[product].place_buy_order(
            #             # math.floor(prices[product] - self.spreads[product]),
            #             self.managers[product].get_worst_sell_order()[0],
            #             quantities[product],
            #         )
        # Price diff is low, long the pair by longing the basket
        # and shorting the components
        # elif price_diff < self.mean_diff - self.trade_signal:
        elif price_diff < -std * 0.45:
            quantities = self.calc_order_quantities(positions, long_pair=True)
            if quantities[GIFT_BASKET] > 0:
                self.managers[GIFT_BASKET].place_buy_order(
                    # math.floor(prices[GIFT_BASKET] - self.spreads[GIFT_BASKET]),
                    self.managers[GIFT_BASKET].get_worst_sell_order()[0],
                    quantities[GIFT_BASKET],
                )
            # for product in BASKET_COMPOSITION:
            #     if quantities[product] < 0:
            #         self.managers[product].place_sell_order(
            #             # math.ceil(prices[product] + self.spreads[product]),
            #             self.managers[product].get_worst_buy_order()[0],
            #             quantities[product],
            #         )

    def run_strawberry(self) -> None:
        self.berry_data_dim=25
        self.berry_position_open=-0.06306734329308888 + 0.05 * 1.0535987849332602
        self.berry_position_close=-0.06306734329308888 + 0.05 * 1.0535987849332602
        self.berry_trade_amount=100
        self.berry_price_diff=1

        prices = {
            product: manager.get_VWAP() for product, manager in self.managers.items()
        }
        positions = {
            product: manager.get_position()
            for product, manager in self.managers.items()
        }

        running_avg = None
        trader_data = self.managers[STRAWBERRIES].trader_data
        berry_data = trader_data.get("berry_data", None)
        if berry_data is None:
            berry_data = []
        elif len(berry_data) == self.berry_data_dim:
            running_avg = sum(berry_data) / self.berry_data_dim
            berry_data = berry_data[1:]
        berry_data.append(prices[STRAWBERRIES])
        self.managers[STRAWBERRIES].add_trader_data("berry_data", berry_data)

        if running_avg is None:
            return

        if prices[STRAWBERRIES] - running_avg > self.berry_position_open:
            quantity = min(
                self.berry_trade_amount, self.managers[STRAWBERRIES].max_buy_amount()
            )
            if quantity > 0:
                self.managers[STRAWBERRIES].place_buy_order(
                    prices[STRAWBERRIES] - self.berry_price_diff, quantity
                )
        elif (
            positions[STRAWBERRIES] > 0
            and prices[STRAWBERRIES] - running_avg < self.berry_position_close
        ):
            quantity = -positions[STRAWBERRIES]
            buy_orders = self.managers[STRAWBERRIES].get_buy_orders()
            worst_price = next(reversed(buy_orders))
            self.managers[STRAWBERRIES].place_sell_order(worst_price, quantity)
        elif prices[STRAWBERRIES] - running_avg < -self.berry_position_open:
            quantity = max(
                -self.berry_trade_amount, self.managers[STRAWBERRIES].max_sell_amount()
            )
            if quantity < 0:
                self.managers[STRAWBERRIES].place_sell_order(
                    prices[STRAWBERRIES] + self.berry_price_diff, quantity
                )
        elif (
            positions[STRAWBERRIES] < 0
            and prices[STRAWBERRIES] - running_avg > -self.berry_position_close
        ):
            quantity = -positions[STRAWBERRIES]
            sell_orders = self.managers[STRAWBERRIES].get_sell_orders()
            worst_price = next(iter(sell_orders))
            self.managers[STRAWBERRIES].place_buy_order(worst_price, quantity)
    
    def run_chocolate_rose(self, state: TradingState) -> None:
        prices = {
            product: manager.get_VWAP() for product, manager in self.managers.items()
        }
        positions = {
            product: manager.get_position()
            for product, manager in self.managers.items()
        }

        if prices[CHOCOLATE] is None or prices[ROSES] is None:
            return
        price_diff = prices[ROSES] - (prices[CHOCOLATE] * 1.342728 + 3878.738377)

        std = 90.9085376833297
        # Price diff is high, short the pair by shorting roses and longing chocolate
        if price_diff > std * 0.42:
            sell_quantity = self.managers[ROSES].max_sell_amount()
            if sell_quantity < 0:
                self.managers[ROSES].place_sell_order(
                    self.managers[ROSES].get_worst_buy_order()[0],
                    sell_quantity
                )
            buy_quantity = self.managers[CHOCOLATE].max_buy_amount()
            if buy_quantity > 0:
                self.managers[CHOCOLATE].place_buy_order(
                    self.managers[CHOCOLATE].get_worst_sell_order()[0],
                    buy_quantity
                )
        # Price diff is low, long the pair by longing roses and shorting chocolate
        elif price_diff < -std * 0.42:
            buy_quantity = self.managers[ROSES].max_buy_amount()
            if buy_quantity > 0:
                self.managers[ROSES].place_buy_order(
                    self.managers[ROSES].get_worst_sell_order()[0],
                    buy_quantity
                )
            sell_quantity = self.managers[CHOCOLATE].max_sell_amount()
            if sell_quantity < 0:
                self.managers[CHOCOLATE].place_sell_order(
                    self.managers[CHOCOLATE].get_worst_buy_order()[0],
                    sell_quantity
                )


    def run(self, state: TradingState) -> None:
        self.run_basket(state)
        self.run_strawberry()
        self.run_chocolate_rose(state)


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
            Listing(symbol=ORCHIDS, product=ORCHIDS, denomination=SEASHELLS),
            arb_margin=1.2,
            manager=managers[ORCHIDS],
        )
        round_3_products = [CHOCOLATE, STRAWBERRIES, ROSES, GIFT_BASKET]
        basket_pair_configs = BasketPairConfigs(
            managers={product: managers[product] for product in round_3_products},
            mean_diff=379.486,
            trade_signal=76.413 * 1.45,
            max_basket_position=60,
            basket_order_quantity=58,
            spreads={product: 1 for product in round_3_products},
        )

        # initialize traders
        amethyst_trader = AmethystTrader(amethyst_configs)
        starfruit_trader = StarfruitTrader(starfruit_configs)
        orchid_trader = OrchidTrader(orchid_configs)
        basket_pair_trader = BasketPairTrader(basket_pair_configs)

        # run traders
        amethyst_trader.run(state)
        starfruit_trader.run(state)
        orchid_trader.run(state)
        basket_pair_trader.run(state)

        # create orders, conversions and trader_data
        orders = {}
        conversions = 0
        new_trader_data = {}

        orders[AMETHYSTS] = amethyst_trader.manager.pending_orders()
        orders[STARFRUIT] = starfruit_trader.manager.pending_orders()
        orders[ORCHIDS] = orchid_trader.manager.pending_orders()
        for product in round_3_products:
            orders[product] = basket_pair_trader.managers[product].pending_orders()

        # conversions = managers[ORCHIDS].conversions

        for product in PRODUCTS:
            new_trader_data.update(managers[product].get_new_trader_data())
        new_trader_data = jsonpickle.encode(new_trader_data)

        logger.flush(state, orders, conversions, new_trader_data)
        return orders, conversions, new_trader_data
