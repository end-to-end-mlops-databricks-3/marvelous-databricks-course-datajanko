"""Configuration file for the project."""

import datetime
from typing import Any

import yaml
from pydantic import BaseModel


class ParsingConfig(BaseModel):
    """Represents the configuration for Parsing the data.

    Allows for dynamic renaming, transforming features into Categoricals and parsing a date column
    """

    rename: dict[str, str] | None
    categories: list[str]
    date_column: str


class SelectionConfig(BaseModel):
    """Selects and declares data for model inputs."""

    features: list[str]
    date_column: str
    categories: list[str] | None = None
    target: str


class ProjectConfig(BaseModel):
    """Represent project configuration parameters loaded from YAML.

    Handles feature specifications, catalog details, and experiment parameters.
    Supports environment-specific configuration overrides.
    """

    raw_data_columns: list[str] | None = None
    parsing: ParsingConfig | None = None
    selection: SelectionConfig
    catalog_name: str
    schema_name: str
    last_training_day: datetime.date
    validation_start_day: datetime.date
    parameters: dict[str, Any]
    fit_parameters: dict[str, Any]
    experiment_name_basic: str
    experiment_name_fe: str | None = None

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load and parse configuration settings from a YAML file.

        :param config_path: Path to the YAML configuration file
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initialized with parsed configuration
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]

            return cls(**config_dict)


class Tags(BaseModel):
    """Represents a set of tags for a Git commit.

    Contains information about the Git SHA, branch, and job run ID.
    """

    git_sha: str
    branch: str
