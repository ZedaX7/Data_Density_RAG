"""
Pydantic schemas for structured battery pack design output.

These schemas are used with the Outlines library to enforce valid JSON
output from the LLM at decode time.
"""

from pydantic import BaseModel, Field
from typing import List


class BatteryDesign(BaseModel):
    """Structured output schema for battery pack designs."""

    series_count: int = Field(
        ge=1,
        le=16,
        description="Number of cells connected in series"
    )
    parallel_count: int = Field(
        ge=1,
        le=16,
        description="Number of cells connected in parallel"
    )
    design_voltage: float = Field(
        gt=0,
        description="Nominal pack voltage in volts"
    )
    design_capacity: float = Field(
        gt=0,
        description="Pack capacity in amp-hours"
    )
    design_width: float = Field(
        gt=0,
        description="Pack width in millimeters"
    )
    design_depth: float = Field(
        gt=0,
        description="Pack depth in millimeters"
    )
    design_height: float = Field(
        gt=0,
        description="Pack height in millimeters"
    )
    cell_locations: List[List[int]] = Field(
        description="List of [x, y, z] coordinates for each cell position"
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of design choices"
    )
