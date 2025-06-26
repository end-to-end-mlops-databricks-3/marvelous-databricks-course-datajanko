import pyspark.sql.functions as F
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from mlops_course.common import create_parser
from mlops_course.config import ProjectConfig
from mlops_course.data_processor import attach_generated_data, bootstrap, create_data_split_and_update, observe_outcomes

args = create_parser()
logger.info(args)
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
is_test = args.is_test
is_bootstrap = args.is_bootstrap
max_date = config.max_raw_data_date
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


spark = SparkSession.builder.getOrCreate()
val_offset = config.validation_size_in_days
test_offset = config.test_set_size_in_days


bootstrap(config, is_bootstrap, max_date, spark)  # type:ignore

all_data = spark.read.table(f"{config.catalog_name}.{config.schema_name}.parsed_data")
original_data = all_data.filter(F.col("date") <= F.lit(max_date))

max_date = max(all_data.agg({"date": "max"}).collect()[0][0], max_date)

drift = 0


dists = attach_generated_data(config, max_date, spark, all_data.schema, original_data, drift)


create_data_split_and_update(config, spark, val_offset, test_offset, all_data)


observe_outcomes(spark, all_data, dists)
