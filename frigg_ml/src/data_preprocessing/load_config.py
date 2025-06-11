# pydantic for schema (model) validation

import yaml
from pydantic import BaseModel, Field
from typing import List, Optional


class FeatureConfig(BaseModel):
    """
    Configuration for feature engineering step.
    """

    numerical: List[str] = Field(
        default_factory=list, description="List of numerical features."
    )
    categorical: List[str] = Field(
        default_factory=list, description="List of categorical features."
    )


## STEPS CONFIGURATION
class CategoricalStepsConfig(BaseModel):
    """
    Configuration for categorical feature preprocessing steps.
    """

    imputer: Optional[str] = None
    encoder: Optional[str] = None


class NumericalStepsConfig(BaseModel):
    """
    Configuration for numerical feature preprocessing steps.
    """

    imputer: Optional[str] = None
    scaler: Optional[str] = None


class StepsConfig(BaseModel):
    """
    Configuration for the steps in the data preprocessing pipeline.
    """

    categorical: Optional[CategoricalStepsConfig] = None
    numerical: Optional[NumericalStepsConfig] = None


## FEATURE CONFIGURATION
class PreprocessorConfig(BaseModel):
    """
    Configuration for data preprocessing step.
    """

    features: "FeatureConfig" = Field(..., description="Feature configuration.")
    steps: "StepsConfig" = Field(..., description="Steps configuration.")


def load_config(config_path: str) -> BaseModel:
    """
    Load the configuration from a YAML file

    Args:
       config_path (str): Path to the YAML configuration file.

    Returns:
        BaseModel: Pydantic model containing the configuration.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return PreprocessorConfig(**config)
