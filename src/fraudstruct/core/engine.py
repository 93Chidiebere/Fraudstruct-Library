class FraudstructEngine:
    def __init__(self, df):
        self.df = df
        self.engine_type = None

    @classmethod
    def from_dataframe(cls, df):
        if df.__class__.__name__ == "DataFrame":
            from fraudstruct.engines.pandas import PandasEngine
            return PandasEngine(df)
        else:
            from fraudstruct.engines.spark import SparkEngine
            return SparkEngine(df)

    def rolling_sum(self, group_col, value_col, window):
        raise NotImplementedError

    def rolling_count(self, group_col, window):
        raise NotImplementedError

    def select(self, cols):
        raise NotImplementedError
