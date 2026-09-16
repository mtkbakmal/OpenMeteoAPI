import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Импортируем объекты из вашего файла с приложением (замените `script` на имя вашего файла, если оно другое)
from script import Base, City, Forecast, User, app, get_db

# Тестовая БД в памяти SQLite
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


# ==================== Фикстуры ====================

@pytest_asyncio.fixture(autouse=True)
async def prepare_database():
    """Создаёт таблицы перед каждым тестом и очищает после."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db():
    """Переопределение зависимости get_db для использования тестовой БД."""
    async with TestingSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture
async def client():
    """Асинхронный клиент для выполнения запросов к приложению."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_open_meteo_daily():
    """Тестовые данные суточного прогноза Open-Meteo с шагом в 15 минут."""
    return {
        "minutely_15": {
            "time": [
                "2026-03-31T14:00",
                "2026-03-31T14:15",
                "2026-03-31T14:30",
                "2026-03-31T14:45",
            ],
            "temperature_2m": [15.0, 15.5, 16.0, 16.5],
            "relative_humidity_2m": [60, 58, 55, 53],
            "wind_speed_10m": [3.2, 3.5, 4.0, 4.1],
            "precipitation": [0.0, 0.0, 0.2, 0.0],
        }
    }


# ==================== Тесты ====================

# --- 1. Тесты /weather/current ---

@pytest.mark.asyncio
async def test_get_current_weather_success(client):
    mock_data = {
        "latitude": 55.75,
        "longitude": 37.61,
        "timezone": "Europe/Moscow",
        "time": "2026-03-31T14:00",
        "temperature": 12.5,
        "wind_speed": 4.2,
        "pressure": 1013.2,
    }
    with patch("script.current_weather", new_callable=AsyncMock) as mock_curr:
        mock_curr.return_value = mock_data

        response = await client.get("/weather/current?latitude=55.75&longitude=37.61")
        assert response.status_code == 200
        data = response.json()
        assert data["temperature"] == 12.5
        assert data["timezone"] == "Europe/Moscow"


@pytest.mark.asyncio
async def test_get_current_weather_validation_error(client):
    # Невалидная широта (> 90)
    response = await client.get("/weather/current?latitude=100.0&longitude=37.61")
    assert response.status_code == 422


# --- 2. Тесты /users (Регистрация пользователей) ---

@pytest.mark.asyncio
async def test_create_user_success(client):
    response = await client.post("/users", json={"name": "Alice"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alice"
    assert "id" in data


@pytest.mark.asyncio
async def test_create_user_duplicate_conflict(client):
    await client.post("/users", json={"name": "Alice"})
    # Повторная регистрация с тем же именем
    response = await client.post("/users", json={"name": "Alice"})
    assert response.status_code == 409
    assert response.json()["detail"] == "Такой пользователь уже существует"


# --- 3. Тесты /users/{user_id}/cities (Добавление и получение городов) ---

@pytest.mark.asyncio
async def test_add_city_success(client, mock_open_meteo_daily):
    # Создаем пользователя
    user_resp = await client.post("/users", json={"name": "Bob"})
    user_id = user_resp.json()["id"]

    with patch("script.request_daily_forecast", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_open_meteo_daily

        response = await client.post(
            f"/users/{user_id}/cities",
            json={"name": "Moscow", "latitude": 55.75, "longitude": 37.61},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Moscow"
        assert data["message"] == "Город добавлен"


@pytest.mark.asyncio
async def test_add_city_user_not_found(client):
    response = await client.post(
        "/users/999/cities",
        json={"name": "Moscow", "latitude": 55.75, "longitude": 37.61},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Пользователь не найден"


@pytest.mark.asyncio
async def test_add_duplicate_city_conflict(client, mock_open_meteo_daily):
    user_resp = await client.post("/users", json={"name": "Bob"})
    user_id = user_resp.json()["id"]

    with patch("script.request_daily_forecast", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_open_meteo_daily
        # Добавляем первый раз
        await client.post(
            f"/users/{user_id}/cities",
            json={"name": "Moscow", "latitude": 55.75, "longitude": 37.61},
        )
        # Добавляем второй раз тот же город (регистронезависимо)
        response = await client.post(
            f"/users/{user_id}/cities",
            json={"name": "moscow", "latitude": 55.75, "longitude": 37.61},
        )
        assert response.status_code == 409
        assert response.json()["detail"] == "У вас уже есть город с таким именем"


@pytest.mark.asyncio
async def test_list_user_cities(client, mock_open_meteo_daily):
    user_resp = await client.post("/users", json={"name": "Charlie"})
    user_id = user_resp.json()["id"]

    with patch("script.request_daily_forecast", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_open_meteo_daily
        await client.post(
            f"/users/{user_id}/cities",
            json={"name": "Berlin", "latitude": 52.52, "longitude": 13.40},
        )

    response = await client.get(f"/users/{user_id}/cities")
    assert response.status_code == 200
    cities = response.json()["cities"]
    assert len(cities) == 1
    assert cities[0]["name"] == "Berlin"


# --- 4. Тесты /users/{user_id}/weather/{city_name} (Поиск погоды и ближайшего времени) ---

@pytest.mark.asyncio
async def test_get_city_weather_nearest_time_matching(client, mock_open_meteo_daily):
    """
    Проверяем, что при запросе времени 14:37 возвращается прогноз на 14:30
    (ближайший слот среди 14:00, 14:15, 14:30, 14:45).
    """
    user_resp = await client.post("/users", json={"name": "Dave"})
    user_id = user_resp.json()["id"]

    with patch("script.request_daily_forecast", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_open_meteo_daily
        await client.post(
            f"/users/{user_id}/cities",
            json={"name": "Paris", "latitude": 48.85, "longitude": 2.35},
        )

    # Запрашиваем 14:37 -> должно округлиться/смаппиться на 14:30 (разница 7 минут vs 8 минут до 14:45)
    response = await client.get(
        f"/users/{user_id}/weather/paris?time=14:37&temperature=true&humidity=true"
    )
    assert response.status_code == 200
    data = response.json()

    assert data["city"] == "Paris"
    assert data["time"] == "14:30"  # Найдено ближайшее время
    assert data["temperature"] == 16.0
    assert data["humidity"] == 55


@pytest.mark.asyncio
async def test_get_city_weather_query_filters(client, mock_open_meteo_daily):
    """Проверяем фильтрацию возвращаемых полей через query-параметры."""
    user_resp = await client.post("/users", json={"name": "Eve"})
    user_id = user_resp.json()["id"]

    with patch("script.request_daily_forecast", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_open_meteo_daily
        await client.post(
            f"/users/{user_id}/cities",
            json={"name": "Rome", "latitude": 41.90, "longitude": 12.49},
        )

    # Запрашиваем только температуру и осадки
    response = await client.get(
        f"/users/{user_id}/weather/rome?time=14:00&temperature=true&humidity=false&wind_speed=false&precipitation=true"
    )
    assert response.status_code == 200
    data = response.json()

    assert "temperature" in data
    assert "precipitation" in data
    assert "humidity" not in data
    assert "wind_speed" not in data


@pytest.mark.asyncio
async def test_get_city_weather_invalid_time_format(client):
    user_resp = await client.post("/users", json={"name": "Frank"})
    user_id = user_resp.json()["id"]

    response = await client.get(f"/users/{user_id}/weather/rome?time=25:99")
    assert response.status_code == 400
    assert "Время должно быть в формате ЧАС:МИНУТЫ" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_city_weather_city_not_found(client):
    user_resp = await client.post("/users", json={"name": "Grace"})
    user_id = user_resp.json()["id"]

    response = await client.get(f"/users/{user_id}/weather/unknown_city?time=12:00")
    assert response.status_code == 404
    assert response.json()["detail"] == "Город не найден у данного пользователя"