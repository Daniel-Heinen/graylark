"""API endpoints"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
# Modified 2023-06-29
# Modified 2024-09-18
