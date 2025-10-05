
"""
Comprehensive test suite for app.py
"""
import pytest
import httpx
from unittest.mock import AsyncMock, Mock, patch
import asyncio

from app import (
    get_users_ids_mapping,
    _get_latest_tweets,
    _aggregate_tweets,
    get_aggregated_tweets,
    chunk_list,
    main,
)


# Fixtures
@pytest.fixture
def mock_client():
    """Create a mock httpx.AsyncClient"""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
def sample_tweets():
    """Sample tweet data for testing"""
    return [
        {
            "created_at": "2024-01-01T12:00:00Z",
            "likes": 100,
            "retweets": 50,
        },
        {
            "created_at": "2024-01-02T12:00:00Z",
            "likes": 200,
            "retweets": 75,
        },
        {
            "created_at": "2024-01-03T12:00:00Z",
            "likes": 150,
            "retweets": 60,
        },
    ]


@pytest.fixture
def sample_api_response():
    """Sample API response for tweets"""
    return {
        "data": [
            {
                "created_at": "2024-01-01T12:00:00Z",
                "public_metrics": {
                    "like_count": 100,
                    "retweet_count": 50,
                },
            },
            {
                "created_at": "2024-01-02T12:00:00Z",
                "public_metrics": {
                    "like_count": 200,
                    "retweet_count": 75,
                },
            },
        ]
    }


# Tests for get_users_ids_mapping
@pytest.mark.asyncio
async def test_get_users_ids_mapping_success(mock_client):
    """Test successful user ID mapping retrieval"""
    mock_response = Mock()
    mock_response.json.return_value = {
        "data": [
            {"id": "123", "username": "user1"},
            {"id": "456", "username": "user2"},
        ]
    }
    mock_response.raise_for_status = Mock()
    mock_client.get.return_value = mock_response

    usernames = ["user1", "user2"]
    result = await get_users_ids_mapping(mock_client, usernames)

    assert result == {"123": "user1", "456": "user2"}
    mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_get_users_ids_mapping_empty_list(mock_client):
    """Test user ID mapping with empty username list"""
    mock_response = Mock()
    mock_response.json.return_value = {"data": []}
    mock_response.raise_for_status = Mock()
    mock_client.get.return_value = mock_response

    result = await get_users_ids_mapping(mock_client, [])

    assert result == {}


@pytest.mark.asyncio
async def test_get_users_ids_mapping_api_error(mock_client):
    """Test user ID mapping when API returns error"""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Error", request=Mock(), response=Mock()
    )
    mock_client.get.return_value = mock_response

    with pytest.raises(httpx.HTTPStatusError):
        await get_users_ids_mapping(mock_client, ["user1"])


# Tests for _get_latest_tweets
@pytest.mark.asyncio
async def test_get_latest_tweets_success(mock_client, sample_api_response):
    """Test successful tweet retrieval"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_api_response
    mock_response.raise_for_status = Mock()
    mock_client.get.return_value = mock_response

    result = await _get_latest_tweets(mock_client, "123")

    assert len(result) == 2
    assert result[0]["created_at"] == "2024-01-01T12:00:00Z"
    assert result[0]["likes"] == 100
    assert result[0]["retweets"] == 50


@pytest.mark.asyncio
async def test_get_latest_tweets_user_not_found(mock_client):
    """Test tweet retrieval when user is not found (404)"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_client.get.return_value = mock_response

    result = await _get_latest_tweets(mock_client, "999")

    assert result == []


@pytest.mark.asyncio
async def test_get_latest_tweets_no_data(mock_client):
    """Test tweet retrieval when user has no tweets"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_response.raise_for_status = Mock()
    mock_client.get.return_value = mock_response

    result = await _get_latest_tweets(mock_client, "123")

    assert result == []


@pytest.mark.asyncio
async def test_get_latest_tweets_with_custom_limit(mock_client, sample_api_response):
    """Test tweet retrieval with custom limit parameter"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_api_response
    mock_response.raise_for_status = Mock()
    mock_client.get.return_value = mock_response

    await _get_latest_tweets(mock_client, "123", limit=5)

    call_args = mock_client.get.call_args
    assert call_args[1]["params"]["max_results"] == 5


# Tests for _aggregate_tweets
def test_aggregate_tweets_with_data(sample_tweets):
    """Test tweet aggregation with valid data"""
    result = _aggregate_tweets(sample_tweets)

    assert result["total_tweets"] == 3
    assert result["total_likes"] == 450
    assert result["total_retweets"] == 185
    assert result["latest_tweet"] == "2024-01-03T12:00:00Z"


def test_aggregate_tweets_empty_list():
    """Test tweet aggregation with empty list"""
    result = _aggregate_tweets([])

    assert result["total_tweets"] == 0
    assert result["total_likes"] == 0
    assert result["total_retweets"] == 0
    assert result["latest_tweet"] is None


def test_aggregate_tweets_single_tweet():
    """Test tweet aggregation with single tweet"""
    tweets = [
        {
            "created_at": "2024-01-01T12:00:00Z",
            "likes": 100,
            "retweets": 50,
        }
    ]
    result = _aggregate_tweets(tweets)

    assert result["total_tweets"] == 1
    assert result["total_likes"] == 100
    assert result["total_retweets"] == 50
    assert result["latest_tweet"] == "2024-01-01T12:00:00Z"


def test_aggregate_tweets_zero_metrics():
    """Test tweet aggregation with zero likes and retweets"""
    tweets = [
        {
            "created_at": "2024-01-01T12:00:00Z",
            "likes": 0,
            "retweets": 0,
        }
    ]
    result = _aggregate_tweets(tweets)

    assert result["total_tweets"] == 1
    assert result["total_likes"] == 0
    assert result["total_retweets"] == 0


# Tests for get_aggregated_tweets
@pytest.mark.asyncio
async def test_get_aggregated_tweets_success(mock_client, sample_api_response):
    """Test full aggregation flow"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_api_response
    mock_response.raise_for_status = Mock()
    mock_client.get.return_value = mock_response

    result = await get_aggregated_tweets(mock_client, "123")

    assert result["total_tweets"] == 2
    assert result["total_likes"] == 300
    assert result["total_retweets"] == 125


@pytest.mark.asyncio
async def test_get_aggregated_tweets_semaphore():
    """Test that semaphore is used correctly"""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}
    mock_response.raise_for_status = Mock()
    mock_client.get.return_value = mock_response

    # Run multiple concurrent requests
    tasks = [get_aggregated_tweets(mock_client, f"user_{i}") for i in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10


@pytest.mark.asyncio
async def test_get_aggregated_tweets_respects_semaphore_limit():
    """Test that semaphore actually limits concurrent requests"""
    from app import SEMAPHORE, MAX_CONCURRENT_REQUESTS

    mock_client = AsyncMock()
    concurrent_count = 0
    max_concurrent = 0

    async def mock_get(*args, **kwargs):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.1)  # Simulate API delay
        concurrent_count -= 1

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        return mock_response

    mock_client.get = mock_get

    # Run more tasks than semaphore limit
    tasks = [get_aggregated_tweets(mock_client, f"user_{i}") for i in range(20)]
    await asyncio.gather(*tasks)

    # Verify we never exceeded the limit
    assert max_concurrent <= MAX_CONCURRENT_REQUESTS


# Tests for chunk_list
def test_chunk_list_normal():
    """Test chunking a list into equal parts"""
    lst = list(range(25))
    result = chunk_list(lst, size=10)

    assert len(result) == 3
    assert result[0] == list(range(10))
    assert result[1] == list(range(10, 20))
    assert result[2] == list(range(20, 25))


def test_chunk_list_exact_size():
    """Test chunking when list size is exact multiple of chunk size"""
    lst = list(range(20))
    result = chunk_list(lst, size=10)

    assert len(result) == 2
    assert len(result[0]) == 10
    assert len(result[1]) == 10


def test_chunk_list_smaller_than_chunk():
    """Test chunking when list is smaller than chunk size"""
    lst = [1, 2, 3]
    result = chunk_list(lst, size=10)

    assert len(result) == 1
    assert result[0] == [1, 2, 3]


def test_chunk_list_empty():
    """Test chunking an empty list"""
    result = chunk_list([], size=10)

    assert result == []


def test_chunk_list_size_one():
    """Test chunking with size 1"""
    lst = [1, 2, 3]
    result = chunk_list(lst, size=1)

    assert len(result) == 3
    assert result == [[1], [2], [3]]


# Tests for main
@pytest.mark.asyncio
async def test_main_success():
    """Test main function with successful flow"""
    with patch("app.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock get_users_ids_mapping
        mock_users_response = Mock()
        mock_users_response.json.return_value = {
            "data": [
                {"id": "123", "username": "user1"},
                {"id": "456", "username": "user2"},
            ]
        }
        mock_users_response.raise_for_status = Mock()

        # Mock get_latest_tweets
        mock_tweets_response = Mock()
        mock_tweets_response.status_code = 200
        mock_tweets_response.json.return_value = {
            "data": [
                {
                    "created_at": "2024-01-01T12:00:00Z",
                    "public_metrics": {"like_count": 100, "retweet_count": 50},
                }
            ]
        }
        mock_tweets_response.raise_for_status = Mock()

        mock_client.get.side_effect = [
            mock_users_response,
            mock_tweets_response,
            mock_tweets_response,
        ]

        result = await main(["user1", "user2"])

        assert "user1" in result
        assert "user2" in result
        assert result["user1"]["total_tweets"] == 1
        assert result["user2"]["total_tweets"] == 1


@pytest.mark.asyncio
async def test_main_empty_usernames():
    """Test main function with empty username list"""
    result = await main([])

    assert result == {}


@pytest.mark.asyncio
async def test_main_large_username_list():
    """Test main function with large username list (chunking)"""
    with patch("app.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Create 150 usernames to test chunking
        usernames = [f"user{i}" for i in range(150)]

        # Mock responses
        def create_user_response(start, end):
            mock_response = Mock()
            mock_response.json.return_value = {
                "data": [
                    {"id": str(i), "username": f"user{i}"}
                    for i in range(start, end)
                ]
            }
            mock_response.raise_for_status = Mock()
            return mock_response

        mock_tweet_response = Mock()
        mock_tweet_response.status_code = 200
        mock_tweet_response.json.return_value = {"data": []}
        mock_tweet_response.raise_for_status = Mock()

        # Setup responses for 2 chunks (100 + 50)
        responses = [
            create_user_response(0, 100),
            *[mock_tweet_response] * 100,
            create_user_response(100, 150),
            *[mock_tweet_response] * 50,
        ]
        mock_client.get.side_effect = responses

        result = await main(usernames)

        assert len(result) == 150


@pytest.mark.asyncio
async def test_main_with_bearer_token():
    """Test that main function uses bearer token from environment"""
    with patch("app.httpx.AsyncClient") as mock_client_class, \
         patch("app.BEARER", "test_token"):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response

        await main([])

        # Verify client was created with correct headers
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test_token"


@pytest.mark.asyncio
async def test_main_integration():
    """Integration test with real API structure (mocked responses)"""
    with patch("app.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Simulate complete flow with realistic data
        mock_users_response = Mock()
        mock_users_response.json.return_value = {
            "data": [
                {"id": "12345", "username": "elonmusk"},
                {"id": "67890", "username": "naval"},
            ]
        }
        mock_users_response.raise_for_status = Mock()

        mock_tweets_elon = Mock()
        mock_tweets_elon.status_code = 200
        mock_tweets_elon.json.return_value = {
            "data": [
                {
                    "created_at": "2024-01-03T10:00:00Z",
                    "public_metrics": {"like_count": 5000, "retweet_count": 1000},
                },
                {
                    "created_at": "2024-01-02T10:00:00Z",
                    "public_metrics": {"like_count": 3000, "retweet_count": 500},
                },
            ]
        }
        mock_tweets_elon.raise_for_status = Mock()

        mock_tweets_naval = Mock()
        mock_tweets_naval.status_code = 200
        mock_tweets_naval.json.return_value = {
            "data": [
                {
                    "created_at": "2024-01-01T10:00:00Z",
                    "public_metrics": {"like_count": 2000, "retweet_count": 300},
                },
            ]
        }
        mock_tweets_naval.raise_for_status = Mock()

        mock_client.get.side_effect = [
            mock_users_response,
            mock_tweets_elon,
            mock_tweets_naval,
        ]

        result = await main(["elonmusk", "naval"])

        # Verify complete aggregation
        assert result["elonmusk"]["total_tweets"] == 2
        assert result["elonmusk"]["total_likes"] == 8000
        assert result["elonmusk"]["total_retweets"] == 1500
        assert result["elonmusk"]["latest_tweet"] == "2024-01-03T10:00:00Z"

        assert result["naval"]["total_tweets"] == 1
        assert result["naval"]["total_likes"] == 2000
        assert result["naval"]["total_retweets"] == 300


@pytest.mark.asyncio
async def test_main_output_format():
    """Test that output matches expected JSON structure"""
    with patch("app.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_users_response = Mock()
        mock_users_response.json.return_value = {
            "data": [{"id": "123", "username": "testuser"}]
        }
        mock_users_response.raise_for_status = Mock()

        mock_tweets_response = Mock()
        mock_tweets_response.status_code = 200
        mock_tweets_response.json.return_value = {
            "data": [
                {
                    "created_at": "2024-01-01T12:00:00Z",
                    "public_metrics": {"like_count": 100, "retweet_count": 50},
                }
            ]
        }
        mock_tweets_response.raise_for_status = Mock()

        mock_client.get.side_effect = [mock_users_response, mock_tweets_response]

        result = await main(["testuser"])

        # Verify exact structure matches task requirements
        assert "testuser" in result
        user_data = result["testuser"]
        assert set(user_data.keys()) == {
            "total_tweets",
            "total_likes",
            "total_retweets",
            "latest_tweet",
        }
        assert isinstance(user_data["total_tweets"], int)
        assert isinstance(user_data["total_likes"], int)
        assert isinstance(user_data["total_retweets"], int)
        assert isinstance(user_data["latest_tweet"], str)
