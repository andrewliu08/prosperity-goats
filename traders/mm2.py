import json
import numpy as np
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
    
    def position_adjustment(self, adjustments: list[int], position: int):
        lim = POSITION_LIMITS[self.product]
        cutoffs = np.linspace(-lim, lim, len(adjustments) + 1)
        for adj, cutoff in zip(adjustments, cutoffs[1:-1]):
            if position <= cutoff:
                return adj
        return adjustments[-1]
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = []
        conversions = 0
        trader_data = ""
        buy_book = [12,8]
        #best split so far was like[6,7,7]
        #best so far was [20], adj  =0, 14.7k, gets 1.29k on official
        #[10, 10] adj = 0, gets 14.5k
        #[15,5] adj = 0, gets 14.7k

        #[20], spread = 1, adj = linear, gets 14.9 across 2 days, gets 1.15k on official
            #adj is [-2,-2,-1,-1,0]
        #[20], spread = 1, adj = linear, gets 23k across 3 days, gets 1.16k on official
            #adj is [-2,-1,-1,0]
        #[10,10], spread = 1, adj = linear, gets 23k across 3 days, gets 1.16k on official
            #adj is [-2,-1,-1,0]
        
        #[10,10,15], spread = 1, adj = linear, gets 23.2k across 3 days, gets 1.17k on official
            #adj is [-2,-1,-1,-1,0]
        #[10,10,15], spread = 1, adj = linear, gets 23.2k across 3 days, gets 1.17k on official
            #adj is [-2,-1,-1,-1,0,0]
        #[20], spread = 1, adj = linear, gets 15.5k across 2 days, gets 1.17k on official
            #adj is [-2,-1,-1,-1,0,0]
        #[20], spread = 1, adj = linear, gets 15.6k across 2 days, gets 1.19k on official
            #adj is [-1,-1,-1,0,0]
        #[12,8] is also a promising split
                

        position = state.position.get(self.product, 0)
        limit = POSITION_LIMITS[self.product]
        buy_quota = limit-position
        sell_quota = -limit-position
        spread = 1

        adj = self.position_adjustment([-1,-1,-1,0,1,1,1],position)
        bp = 10000 - spread - adj
        sp = 10000 + spread - adj
        

        for i,qty in enumerate (buy_book):
            q = min(qty, buy_quota)
            orders.append(Order(self.product, bp-i,q))
            buy_quota-=q
        if buy_quota > 0:
            orders.append(Order(self.product,bp-len(buy_book)+1,buy_quota))


        for i,qty in enumerate (buy_book):
            q = max(-qty,sell_quota)
            orders.append(Order(self.product, sp+i,q))
            sell_quota-=q
        if sell_quota < 0:
            orders.append(Order(self.product,sp+len(buy_book)-1,sell_quota))
        return orders, conversions, trader_data


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # initialize configs
        amethyst_configs = AmethystConfigs(
            Listing(symbol=AMETHYSTS, product=AMETHYSTS, denomination=SEASHELLS),
            price=10_000,
            mm_spread=4,
            quantity=5,
        )

        # initialize traders
        amethyst_trader = AmethystTrader(amethyst_configs)

        # run traders
        amethyst_orders, amethyst_conversions, amethyst_trader_data = (
            amethyst_trader.run(state)
        )

        # create orders, conversions and trader_data
        orders = {}
        conversions = 0
        trader_data = ""

        orders[AMETHYSTS] = amethyst_orders
        conversions += amethyst_conversions
        trader_data += amethyst_trader_data

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
