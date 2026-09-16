# Imports
import asyncio
import json
import httpx
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, status, Query, Depends
from pydantic import BaseModel, Field

from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, UniqueConstraint, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

# Constants
DB_NAME = "weather.db"
DATABASE_URL = f"sqlite+aiosqlite:///./{DB_NAME}"
URL = "https://api.open-meteo.com/v1/forecast"
UPDATE_INTERVAL = 15


# ==================== ORM Models ====================
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)

    # Связь: один пользователь -> много городов
    cities: Mapped[list["City"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class City(Base):
    __tablename__ = "cities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

    user: Mapped["User"] = relationship(back_populates="cities")
    forecasts: Mapped[list["Forecast"]] = relationship(
        back_populates="city", cascade="all, delete-orphan"
    )

    # Уникальность имени города только в рамках одного пользователя
    __table_args__ = (UniqueConstraint("user_id", "name"),)


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(Integer, ForeignKey("cities.id"), nullable=False)
    forecast_date: Mapped[str] = mapped_column(String, nullable=False)
    data: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    city: Mapped["City"] = relationship(back_populates="forecasts")

    __table_args__ = (UniqueConstraint("city_id", "forecast_date"),)


# ==================== DB Setup ====================
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with async_session() as session:
        yield session


# ==================== Schemas ====================
class CityCreate(BaseModel):
    name: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class UserCreate(BaseModel):
    name: str = Field(..., min_length=1)


# ==================== API Requests ====================
async def current_weather(latitude: float, longitude: float):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(["temperature_2m", "wind_speed_10m", "surface_pressure"]),
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(URL, params=params)

    response.raise_for_status()
    data = response.json()

    return {
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "timezone": data["timezone"],
        "time": data["current"]["time"],
        "temperature": data["current"]["temperature_2m"],
        "wind_speed": data["current"]["wind_speed_10m"],
        "pressure": data["current"]["surface_pressure"],
    }


async def request_daily_forecast(latitude: float, longitude: float):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "minutely_15": ",".join(
            ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
        ),
        "forecast_days": 1,
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(URL, params=params)

    response.raise_for_status()
    return response.json()


# ==================== Helpers ====================
def forecast_is_fresh(updated_at: datetime) -> bool:
    now = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    minutes_passed = (now - updated_at).total_seconds() / 60
    return minutes_passed < UPDATE_INTERVAL


# ==================== CRUD ====================
# -- Users --
async def create_user(session: AsyncSession, name: str) -> int:
    user = User(name=name)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user.id


async def get_user_by_id(session: AsyncSession, user_id: int) -> User | None:
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


# -- Cities --
async def create_city(
    session: AsyncSession, user_id: int, name: str, latitude: float, longitude: float
) -> int:
    city = City(
        user_id=user_id,
        name=name.lower(),
        latitude=latitude,
        longitude=longitude,
    )
    session.add(city)
    await session.commit()
    await session.refresh(city)
    return city.id


async def get_city_by_name(
    session: AsyncSession, user_id: int, city_name: str
) -> City | None:
    result = await session.execute(
        select(City).where(City.user_id == user_id, City.name == city_name.lower())
    )
    return result.scalar_one_or_none()


async def get_user_cities(session: AsyncSession, user_id: int) -> list[City]:
    result = await session.execute(
        select(City).where(City.user_id == user_id).order_by(City.id)
    )
    return list(result.scalars().all())


async def get_all_cities(session: AsyncSession) -> list[City]:
    result = await session.execute(select(City))
    return list(result.scalars().all())


# -- Forecasts --
async def fetch_latest_forecast(session: AsyncSession, city_id: int) -> Forecast | None:
    stmt = (
        select(Forecast)
        .where(Forecast.city_id == city_id)
        .order_by(Forecast.updated_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def save_forecast(
    session: AsyncSession, city_id: int, forecast_date: str, forecast: list[dict]
):
    now = datetime.now(timezone.utc)
    data = json.dumps(forecast)

    stmt = sqlite_insert(Forecast).values(
        city_id=city_id, forecast_date=forecast_date, data=data, updated_at=now
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["city_id", "forecast_date"],
        set_={"data": data, "updated_at": now},
    )

    await session.execute(stmt)
    await session.commit()


async def update_city_forecast(
    session: AsyncSession, city_id: int, latitude: float, longitude: float
):
    data = await request_daily_forecast(latitude, longitude)

    minutely = data["minutely_15"]
    forecast = [
        {
            "time": time,
            "temperature": minutely["temperature_2m"][i],
            "humidity": minutely["relative_humidity_2m"][i],
            "wind_speed": minutely["wind_speed_10m"][i],
            "precipitation": minutely["precipitation"][i],
        }
        for i, time in enumerate(minutely["time"])
    ]

    if not forecast:
        raise ValueError("Open-Meteo не вернул прогноз")

    forecast_date = forecast[0]["time"][:10]
    await save_forecast(session, city_id, forecast_date, forecast)


async def get_city_forecast(
    session: AsyncSession, city: City, force_refresh: bool = False
):
    forecast = await fetch_latest_forecast(session, city.id)

    need_refresh = (
        force_refresh or forecast is None or not forecast_is_fresh(forecast.updated_at)
    )

    if need_refresh:
        await update_city_forecast(session, city.id, city.latitude, city.longitude)
        forecast = await fetch_latest_forecast(session, city.id)

    if forecast is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить прогноз",
        )

    return json.loads(forecast.data)


# ==================== Background Updater ====================
async def update_all_forecasts():
    while True:
        async with async_session() as session:
            # Обновляем прогнозы для городов ВСЕХ пользователей
            cities = await get_all_cities(session)

            for city in cities:
                try:
                    await update_city_forecast(session, city.id, city.latitude, city.longitude)
                    print(f"Прогноз обновлён: {city.name}")
                except Exception as error:
                    print(f"Ошибка обновления {city.name}: {error}")
                    await session.rollback()

        await asyncio.sleep(UPDATE_INTERVAL * 60)


# ==================== Lifespan ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    task = asyncio.create_task(update_all_forecasts())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ==================== FastAPI ====================
app = FastAPI(
    title="Weather API",
    description="REST API для получения информации о погоде",
    version="3.0.0",
    lifespan=lifespan,
)


# ==================== Endpoints ====================

# 1. Global current weather (без привязки к пользователю)
@app.get("/weather/current")
async def get_current_weather(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    try:
        return await current_weather(latitude, longitude)
    except httpx.HTTPError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Ошибка при обращении к Open-Meteo API",
        )


# 2. Регистрация пользователя
@app.post("/users")
async def add_user(user: UserCreate, session: AsyncSession = Depends(get_db)):
    try:
        user_id = await create_user(session, user.name)
        return {"id": user_id, "name": user.name}
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Такой пользователь уже существует",
        )


# 3. Добавление города (для конкретного пользователя)
@app.post("/users/{user_id}/cities")
async def add_city(
    user_id: int,
    city: CityCreate,
    session: AsyncSession = Depends(get_db),
):
    user = await get_user_by_id(session, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден",
        )

    try:
        city_id = await create_city(
            session, user_id, city.name, city.latitude, city.longitude
        )
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="У вас уже есть город с таким именем",
        )

    try:
        await update_city_forecast(session, city_id, city.latitude, city.longitude)
    except Exception as error:
        # Откатываем добавление города, если прогноз не получен
        city_obj = await session.get(City, city_id)
        if city_obj:
            await session.delete(city_obj)
            await session.commit()
        print(f"Ошибка добавления города: {error}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Не удалось получить прогноз из Open-Meteo",
        )

    return {
        "id": city_id,
        "name": city.name,
        "latitude": city.latitude,
        "longitude": city.longitude,
        "message": "Город добавлен",
    }


# 4. Получение списка городов пользователя
@app.get("/users/{user_id}/cities")
async def list_user_cities(user_id: int, session: AsyncSession = Depends(get_db)):
    user = await get_user_by_id(session, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден",
        )

    cities = await get_user_cities(session, user_id)
    return {
        "cities": [
            {
                "id": c.id,
                "name": c.name.title(),
                "latitude": c.latitude,
                "longitude": c.longitude,
            }
            for c in cities
        ]
    }


# 5. Получение погоды для конкретного города пользователя
@app.get("/users/{user_id}/weather/{city_name}")
async def get_city_weather(
    user_id: int,
    city_name: str,
    time: str = Query(..., description="Время в формате ЧАС:МИНУТЫ (например, 14:30)"),
    temperature: bool = Query(True, description="Получить температуру"),
    humidity: bool = Query(True, description="Получить влажность"),
    wind_speed: bool = Query(True, description="Получить скорость ветра"),
    precipitation: bool = Query(True, description="Получить осадки"),
    session: AsyncSession = Depends(get_db),
):
    try:
        req_dt = datetime.strptime(time, "%H:%M")
        req_minutes = req_dt.hour * 60 + req_dt.minute
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Время должно быть в формате ЧАС:МИНУТЫ (например, 14:30)",
        )

    user = await get_user_by_id(session, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден",
        )

    city = await get_city_by_name(session, user_id, city_name)
    if city is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Город не найден у данного пользователя",
        )

    try:
        forecast = await get_city_forecast(session, city)
    except httpx.HTTPError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ошибка при получении прогноза",
        )

    if not forecast:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Данные прогноза отсутствуют",
        )

    # Ищем элемент прогноза с минимальной разницей во времени
    weather = min(
        forecast,
        key=lambda item: abs(
            (int(item["time"][11:13]) * 60 + int(item["time"][14:16])) - req_minutes
        ),
    )

    actual_time = weather["time"][11:16]

    result = {
        "city": city.name.title(),
        "requested_time": req_dt.strftime("%H:%M"),
        "time": actual_time,  # Фактическое ближайшее время прогноза (например, 14:30 при запросе 14:37)
    }

    if temperature:
        result["temperature"] = weather["temperature"]
    if humidity:
        result["humidity"] = weather["humidity"]
    if wind_speed:
        result["wind_speed"] = weather["wind_speed"]
    if precipitation:
        result["precipitation"] = weather["precipitation"]

    return result

# ==================== Run ====================
if __name__ == "__main__":
    uvicorn.run("script:app", host="0.0.0.0", port=8000, reload=True)