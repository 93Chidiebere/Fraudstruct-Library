class FraudstructEngine:
    def __init__(self, df):
        self.df = df
        self.engine_type = None

    @classmethod
    def from_dataframe(cls, df, engine_type=None):
        if engine_type == "graph":
            from fraudstruct.engines.graph import GraphEngine
            return GraphEngine(df)
        elif engine_type == "pandas" or (df.__class__.__name__ == "DataFrame" and engine_type is None):
            from fraudstruct.engines.pandas import PandasEngine
            return PandasEngine(df)
        elif engine_type == "spark":
            from fraudstruct.engines.spark import SparkEngine
            return SparkEngine(df)
        else:
            if "pyspark" in str(type(df)):
                from fraudstruct.engines.spark import SparkEngine
                return SparkEngine(df)
            else:
                from fraudstruct.engines.pandas import PandasEngine
                return PandasEngine(df)

    def rolling_sum(self, group_col, value_col, window):
        raise NotImplementedError

    def rolling_count(self, group_col, window):
        raise NotImplementedError

    def select(self, cols):
        raise NotImplementedError
