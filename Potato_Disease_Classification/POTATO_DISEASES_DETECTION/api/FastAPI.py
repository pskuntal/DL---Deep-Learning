# FastAPI tutorial for beginners

import uvicorn
from fastapi import FastAPI
from enum import Enum

app = FastAPI()


@app.get("/hello/{name}")
async def hello(name):
    return f"Welcome to IIT Kanpur Library, Mr./Ms. {name}!"


class AvailCuisine(str, Enum):

    Indian = "Indian"
    Italian = "Italian"
    American = "American"


food_items = {
    "Indian": ["Rasmalai", "Dosa"],
    "Italian": ["Pizza"],
    "American": ["Grass"]
}


@app.get("/get_items/{cuisine}")
async def get_items(cuisine: AvailCuisine):
    return food_items.get(cuisine)


coupon_code = {
    1: "10%",
    2: "20%",
    3: "30%"
}


@app.get("/get_coupon/{code}")
async def get_items(code: int):
    return {"discount_amount": coupon_code.get(code)}
