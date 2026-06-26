from pyspark.sql import Window
import pyspark.sql.functions as F
from fraudstruct.core.engine import FraudstructEngine

class SparkEngine(FraudstructEngine):
    def __init__(self, df):
        super().__init__(df)
        self.engine_type = "spark"
        self.df = df

    def rolling_sum(self, group_col, value_col, window_seconds):
        w = (
            Window
            .partitionBy(group_col)
            .orderBy(F.col("timestamp").cast("long"))
            .rangeBetween(-window_seconds, 0)
        )
        return self.df.withColumn("rolling_sum", F.sum(value_col).over(w))

    def rolling_count(self, group_col, window_seconds):
        w = (
            Window
            .partitionBy(group_col)
            .orderBy(F.col("timestamp").cast("long"))
            .rangeBetween(-window_seconds, 0)
        )
        return self.df.withColumn("rolling_count", F.count("*").over(w))

    def select(self, cols):
        return self.df.select(cols)
