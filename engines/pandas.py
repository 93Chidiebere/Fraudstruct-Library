import pandas as pd
from fraudstruct.core.engine import FraudstructEngine

class PandasEngine(FraudstructEngine):
    def __init__(self, df):
        super().__init__(df)
        self.engine_type = "pandas"
        self.df = df.sort_values("timestamp")

    def rolling_sum(self, group_col, value_col, window):
        return (
            self.df
            .set_index("timestamp")
            .groupby(group_col)[value_col]
            .rolling(window)
            .sum()
            .reset_index(name="rolling_sum")
        )

    def rolling_count(self, group_col, window):
        return (
            self.df
            .set_index("timestamp")
            .groupby(group_col)
            .rolling(window)
            .size()
            .reset_index(name="rolling_count")
        )

    def select(self, cols):
        return self.df[cols]
