import asyncio
import random
import time

import cv2
import numpy as np
import pytest
import pytest_asyncio
from app.main import app
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def test_image_bytes():
    img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.load
@pytest.mark.asyncio
async def test_concurrent_health_checks(async_client: AsyncClient):
    num_requests = 50

    async def check_health():
        response = await async_client.get("/health")
        assert response.status_code == 200
        return response.json()

    start_time = time.time()
    tasks = [check_health() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    elapsed = end_time - start_time
    requests_per_second = num_requests / elapsed

    print("\nHealth check load test:")
    print(f"  Total requests: {num_requests}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {requests_per_second:.2f} req/s")
    print(f"  Avg response time: {(elapsed / num_requests) * 1000:.2f}ms")

    assert all(r["status"] in ["healthy", "degraded"] for r in results)
    assert requests_per_second > 10


@pytest.mark.load
@pytest.mark.asyncio
async def test_concurrent_root_endpoint(async_client: AsyncClient):
    num_requests = 100

    async def get_root():
        response = await async_client.get("/")
        assert response.status_code == 200
        return response.json()

    start_time = time.time()
    tasks = [get_root() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    elapsed = end_time - start_time
    requests_per_second = num_requests / elapsed

    print("\nRoot endpoint load test:")
    print(f"  Total requests: {num_requests}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {requests_per_second:.2f} req/s")

    assert all("version" in r for r in results)
    assert requests_per_second > 50


@pytest.mark.load
@pytest.mark.asyncio
async def test_concurrent_job_status_queries(async_client: AsyncClient, monkeypatch):
    from unittest.mock import Mock

    from app.detector import ThermalDetector
    from app.main import storage
    from app.processor import ImageProcessor

    mock_detector = Mock(spec=ThermalDetector)
    mock_processor = Mock(spec=ImageProcessor)
    mock_processor.sanitize_filename = Mock(return_value="test.jpg")
    mock_processor.validate_image_format = Mock(return_value=True)
    mock_processor.is_zip_file = Mock(return_value=False)
    mock_processor.load_image = Mock()
    monkeypatch.setattr("app.main.detector", mock_detector)
    monkeypatch.setattr("app.main.processor", mock_processor)

    job_ids = []
    for i in range(10):
        job_id = storage.create_job(name=f"LoadTest_{i}", total_images=1)
        job_ids.append(job_id)

    num_requests = 50

    async def get_job_status(job_id: str):
        response = await async_client.get(f"/api/jobs/{job_id}")
        if response.status_code == 200:
            return response.json()
        return None

    start_time = time.time()
    tasks = [get_job_status(random.choice(job_ids)) for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    elapsed = end_time - start_time
    requests_per_second = num_requests / elapsed
    success_count = sum(1 for r in results if r is not None)

    print("\nJob status queries load test:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful requests: {success_count}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {requests_per_second:.2f} req/s")

    assert success_count > 0
    assert requests_per_second > 20


@pytest.mark.load
@pytest.mark.asyncio
async def test_concurrent_image_uploads(
    async_client: AsyncClient,
    test_image_bytes: bytes,
    temp_storage,
    monkeypatch,
):
    from unittest.mock import Mock

    from app.detector import ThermalDetector
    from app.processor import ImageProcessor

    mock_detector = Mock(spec=ThermalDetector)
    mock_processor = Mock(spec=ImageProcessor)
    mock_processor.sanitize_filename = Mock(return_value="test.jpg")
    mock_processor.validate_image_format = Mock(return_value=True)
    mock_processor.is_zip_file = Mock(return_value=False)
    mock_processor.load_image = Mock()
    monkeypatch.setattr("app.main.detector", mock_detector)
    monkeypatch.setattr("app.main.processor", mock_processor)

    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        num_uploads = 20
        concurrent_limit = 5

        async def upload_image(index: int):
            files = {"files": (f"test_{index}.jpg", test_image_bytes, "image/jpeg")}
            data = {"confidence_threshold": "0.5"}
            response = await async_client.post("/api/upload", files=files, data=data)
            if response.status_code == 200:
                return response.json()
            return None

        start_time = time.time()

        semaphore = asyncio.Semaphore(concurrent_limit)

        async def upload_with_limit(index: int):
            async with semaphore:
                return await upload_image(index)

        tasks = [upload_with_limit(i) for i in range(num_uploads)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        elapsed = end_time - start_time
        success_count = sum(1 for r in results if r is not None)
        uploads_per_second = success_count / elapsed if elapsed > 0 else 0

        print("\nImage upload load test:")
        print(f"  Total uploads: {num_uploads}")
        print(f"  Concurrent limit: {concurrent_limit}")
        print(f"  Successful uploads: {success_count}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {uploads_per_second:.2f} uploads/s")

        assert success_count > 0
    finally:
        app.main.storage = original_storage


@pytest.mark.load
@pytest.mark.asyncio
async def test_mixed_workload(
    async_client: AsyncClient,
    test_image_bytes: bytes,
    temp_storage,
    monkeypatch,
):
    from unittest.mock import Mock

    import app.main
    from app.detector import ThermalDetector
    from app.main import storage
    from app.processor import ImageProcessor

    mock_detector = Mock(spec=ThermalDetector)
    mock_processor = Mock(spec=ImageProcessor)
    mock_processor.sanitize_filename = Mock(return_value="test.jpg")
    mock_processor.validate_image_format = Mock(return_value=True)
    mock_processor.is_zip_file = Mock(return_value=False)
    mock_processor.load_image = Mock()
    monkeypatch.setattr("app.main.detector", mock_detector)
    monkeypatch.setattr("app.main.processor", mock_processor)

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_ids = []
        for i in range(5):
            job_id = storage.create_job(name=f"MixedTest_{i}", total_images=1)
            job_ids.append(job_id)

        async def health_check():
            response = await async_client.get("/health")
            return response.status_code == 200

        async def get_root():
            response = await async_client.get("/")
            return response.status_code == 200

        async def get_job_status(job_id: str):
            response = await async_client.get(f"/api/jobs/{job_id}")
            return response.status_code == 200

        async def upload_image(index: int):
            files = {"files": (f"mixed_{index}.jpg", test_image_bytes, "image/jpeg")}
            data = {"confidence_threshold": "0.5"}
            response = await async_client.post("/api/upload", files=files, data=data)
            return response.status_code == 200

        start_time = time.time()

        tasks = []
        tasks.extend([health_check() for _ in range(20)])
        tasks.extend([get_root() for _ in range(20)])
        tasks.extend([get_job_status(job_ids[i % len(job_ids)]) for i in range(15)])
        tasks.extend([upload_image(i) for i in range(10)])

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        elapsed = end_time - start_time
        total_requests = len(tasks)
        success_count = sum(1 for r in results if r)
        requests_per_second = total_requests / elapsed

        print("\nMixed workload test:")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful requests: {success_count}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {requests_per_second:.2f} req/s")
        print(f"  Success rate: {(success_count / total_requests) * 100:.1f}%")

        assert success_count > total_requests * 0.7
    finally:
        app.main.storage = original_storage


@pytest.mark.load
@pytest.mark.asyncio
async def test_stress_test_health_endpoint(async_client: AsyncClient):
    num_requests = 200
    batch_size = 20

    async def check_health():
        response = await async_client.get("/health")
        return response.status_code == 200

    start_time = time.time()
    success_count = 0
    error_count = 0

    for i in range(0, num_requests, batch_size):
        batch = [check_health() for _ in range(min(batch_size, num_requests - i))]
        results = await asyncio.gather(*batch, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
            elif result:
                success_count += 1
            else:
                error_count += 1

    end_time = time.time()
    elapsed = end_time - start_time
    requests_per_second = num_requests / elapsed

    print("\nStress test - health check:")
    print(f"  Total requests: {num_requests}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {requests_per_second:.2f} req/s")
    print(f"  Success rate: {(success_count / num_requests) * 100:.1f}%")

    assert success_count > num_requests * 0.95
    assert requests_per_second > 30
