"""Data models for location"""
from pydantic import BaseModel

class Location(BaseModel):
    latitude: float
    longitude: float
    confidence: float
# Modified 2024-07-10
# Modified 2025-07-23
