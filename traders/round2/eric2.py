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
import pandas as pd


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

INVENTORY_COST = 0.1


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
            jsonpickle.decode(self.state.traderData) if self.state.traderData else {}
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
        arb_margin: float,
        manager: Manager,
    ):
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
        return INVENTORY_COST * quantity * t

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
        vwap = self.manager.get_VWAP()

        #linear regression

        #retrieve the dataframe from the last state
        if 'orchid_data' in state.traderData:
            data = state.traderData.get('orchid_data')
        else:
            #initialize if it doesn't exist
            allColumns = ['transport_fees', 'export_tariffs','import_tariffs', 'sunlight', 'humidity']
            data = pd.dataFrame(columns=allColumns)
        
        #add the new row of the current timestep
        current_timestep_info = {'transport_fees': transport_fees, 'export_tariffs': export_tariff,'import_tariffs': import_tariff, 'sunlight': sunlight, 'humidity': humidity}
        data = data.append(current_timestep_info, ignore_index=True)

        #only ever keep the necessary 7500 rows of information
        while len(data) > 7500:
            data = data.drop(data.index[0])

        self.manager.add_trader_data('orchid_data',data)
        #dictionary of desired features and their coefficients
        feature_coefficient = {
            'hum_prod_dec': 48.67376,
            'sun_prod_dec': 0,
            'prod_hum_1': 629.3902,
            'prod_sun_1': -79.54414,
            'hum_1': 70218.88,
            'sun_1': 7648.580,
            'prod_hum_2': 1271.859,
            'prod_sun_2': -226.6559,
            'hum_2': -87715.21,
            'sun_2': 3452.863,
            'prod_hum_3': 1079.101,
            'prod_sun_3': -308.7377,
            'hum_3': 1938220,
            'sun_3': -2822.175,
            'prod_hum_4': 923.3421,
            'prod_sun_4': -292.5504,
            'hum_4': -2337665,
            'sun_4': -3052.217,
            'prod_hum_5': 911.7624,
            'prod_sun_5': -343.2527,
            'hum_5': 712721.9,
            'sun_5': 13940.49,
            'prod_hum_6': 1194.891,
            'prod_sun_6': -236.9766,
            'hum_6': -1025548,
            'sun_6': -21352.43,
            'prod_hum_7': 1775.479,
            'prod_sun_7': -212.6642,
            'hum_7': 337239.7,
            'sun_7': 9190.124,
            'prod_hum_8': 754.0357e+02,
            'prod_sun_8': -249065.8,
            'hum_8': 366166.6,
            'sun_8': -16096.62,
            'prod_hum_9': -185.7959,
            'prod_sun_9': 466185.4,
            'hum_9': -1283584,
            'sun_9': 7063.233,
            'prod_hum_10': -438.1893,
            'prod_sun_10': -64.62283,
            'hum_10': 453478.7,
            'sun_10': -2320.270,
            'prod_hum_11': -314.2999,
            'prod_sun_11': 0,
            'hum_11': 243749.6,
            'sun_11': 1349.266,
            'prod_hum_12': -550.1919,
            'prod_sun_12': -31.62999,
            'hum_12': -167669.1,
            'sun_12': 17059.70,
            'prod_hum_13': -529.1117,
            'prod_sun_13': -30.30185,
            'hum_13': 283223.7,
            'sun_13': -6111.550,
            'prod_hum_14': -21.94254,
            'prod_sun_14': 0,
            'hum_14': -1374792,
            'sun_14': 4995.459,
            'prod_hum_15': 100.2936,
            'prod_sun_15': -28.14696,
            'hum_15': 26.68270,
            'sun_15': 0,
        }

        #code to construct all the linear regression features
        data['hum_prod_dec'] = np.where((data["HUMIDITY"] < 60) | (data["HUMIDITY"] > 80), abs(data["HUMIDITY"] - 70) / 5.0 * 2.0 - 4, 0)
        data["sun_prod_dec"] = np.where(data["SUNLIGHT"] < 2500, abs(data["SUNLIGHT"] - 2500) / 60 * 4, 0)
        window_size = 500
        for window in range(15):
            start = window * window_size
            end = start + window_size
            if f'prod_hum_{window+1}' in feature_coefficient:
                data[f'prod_hum_{window+1}'] = data['hum_prod_dec'].shift(start).rolling(window=window_size, min_periods=1).mean()
            if f'prod_sun_{window+1}' in feature_coefficient:
                data[f'prod_sun_{window+1}'] = data['sun_prod_dec'].shift(start).rolling(window=window_size, min_periods=1).mean()
            if f'hum_{window+1}' in feature_coefficient:
                data[f'hum_{window+1}'] = data['HUMIDITY'].shift(start).rolling(window=window_size, min_periods=1).mean()
            if f'sun_{window+1}' in feature_coefficient:
                data[f'sun_{window+1}'] = data['SUNLIGHT'].shift(start).rolling(window=window_size, min_periods=1).mean()

        #get the row that we are currently on, in order to predict, and initialize price
        current_row = len(data) - 1
        predicted_price = 0

        #loop through all features, multiply by coefficient and add to predicted price
        for feature in feature_coefficient:
            predicted_price += data.at[current_row, feature] * feature_coefficient.get(feature)

        # Arbitrage
        if position != 0:
            self.manager.set_conversion(-position)
        else:
            exp_pos = position
            sell_orders = self.manager.get_sell_orders()
            for price, qty in sell_orders.items():
                bid_quantity = min(self.manager.max_buy_amount(exp_pos), -qty)
                inventory_cost = self.calc_inventory_cost(bid_quantity, t=1)
                # buy at price now, sell at conv_bid_price next time_step
                if conv_bid_price - price >= self.arb_margin + inventory_cost:
                    self.manager.place_buy_order(price, bid_quantity)
                    exp_pos += bid_quantity

            # maker order expecting that conv_bid_price won't change much
            inventory_cost = self.calc_inventory_cost(
                self.manager.max_buy_amount(exp_pos), t=1
            )
            self.manager.place_buy_order(
                math.floor(conv_bid_price - self.arb_margin - inventory_cost),
                self.manager.max_buy_amount(exp_pos),
            )

            exp_pos = position
            buy_orders = self.manager.get_buy_orders()
            for price, qty in buy_orders.items():
                ask_quantity = max(self.manager.max_sell_amount(exp_pos), -qty)
                inventory_cost = self.calc_inventory_cost(ask_quantity, t=1)
                # sell at price now, buy at conv_ask_price next time_step
                if price - conv_ask_price >= self.arb_margin + inventory_cost:
                    self.manager.place_sell_order(price, ask_quantity)
                    exp_pos += ask_quantity

            # maker order expecting that conv_ask_price won't change much
            inventory_cost = self.calc_inventory_cost(
                self.manager.max_sell_amount(exp_pos), t=1
            )
            self.manager.place_sell_order(
                math.ceil(conv_ask_price + self.arb_margin + inventory_cost),
                self.manager.max_sell_amount(exp_pos),
            )


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

        # orders[AMETHYSTS] = amethyst_trader.manager.pending_orders()
        # orders[STARFRUIT] = starfruit_trader.manager.pending_orders()
        orders[ORCHIDS] = orchid_trader.manager.pending_orders()

        conversions = managers[ORCHIDS].conversions

        for product in PRODUCTS:
            new_trader_data.update(managers[product].get_new_trader_data())
        new_trader_data = jsonpickle.encode(new_trader_data)

        logger.flush(state, orders, conversions, new_trader_data)
        return orders, conversions, new_trader_data
