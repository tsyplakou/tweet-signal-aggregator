import os
import asyncio
import httpx
import json
import logging
from datetime import datetime, timezone

API_BASE = "https://api.twitter.com/2"
BEARER = os.getenv("TWITTER_BEARER_TOKEN")
MAX_CONCURRENT_REQUESTS = 5
USERS_BATCH_SIZE = 100
TWEET_LIMIT = 10
REQUEST_TIMEOUT = 10

SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


class JsonFormatter(logging.Formatter):
    """Logger for Kibana"""

    def format(self, record):
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.msg,
        }
        if record.args:
            log.update(record.args)
        return json.dumps(log, ensure_ascii=False)

logger = logging.getLogger("aggregator")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# TODO: caching/db-storage for users_ids_mapping
# TODO: mark username as INVALID is API didn't return id for username
async def get_users_ids_mapping(client, usernames_batch: list):
    url = f"{API_BASE}/users/by"
    params = {
        "user.fields": "id",
        "usernames": usernames_batch,
    }
    r = await client.get(url, params=params)

    logger.info("fetch_user_ids", {"url": url, "status_code": r.status_code, "count": len(usernames_batch)})

    r.raise_for_status()

    return {
        user_data.get("id"): user_data.get("username")
        for user_data in r.json()["data"]
    }


async def _get_latest_tweets(client, user_id, limit=TWEET_LIMIT):
    url = f"{API_BASE}/users/{user_id}/tweets"
    params = {
        "max_results": limit,
        "tweet.fields": "created_at,public_metrics",
    }
    r = await client.get(url, params=params)
    logger.info("fetch_tweets", {"user_id": user_id, "status_code": r.status_code})

    if r.status_code == 404:  # TODO: if user account was deleted - mark user as deleted
        logger.warning("user_not_found", {"user_id": user_id})
        return []

    r.raise_for_status()

    data = r.json().get("data", [])

    return [
        {
            "created_at": t["created_at"],
            "likes": t["public_metrics"]["like_count"],
            "retweets": t["public_metrics"]["retweet_count"],
        }
        for t in data
    ]


def _aggregate_tweets(tweets):
    if not tweets:
        return {
            "total_tweets": 0,
            "total_likes": 0,
            "total_retweets": 0,
            "latest_tweet": None,
        }
    return {
        "total_tweets": len(tweets),
        "total_likes": sum(t["likes"] for t in tweets),
        "total_retweets": sum(t["retweets"] for t in tweets),
        "latest_tweet": max(t["created_at"] for t in tweets),
    }


async def get_aggregated_tweets(client, user_id):
    async with SEMAPHORE:
        tweets = await _get_latest_tweets(client, user_id)
        aggregated_data = _aggregate_tweets(tweets)
        logger.info("aggregated_user", {"user_id": user_id, **aggregated_data})
        return aggregated_data


def chunk_list(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]


async def main(usernames):
    result = {}
    headers = {"Authorization": f"Bearer {BEARER}"}

    async with httpx.AsyncClient(headers=headers, timeout=REQUEST_TIMEOUT) as client:
        for usernames_chunk in chunk_list(usernames, USERS_BATCH_SIZE): # TODO: dig deeper, to optimize chunk size (twitter API speed)
            users_ids_mapping = await get_users_ids_mapping(client, usernames_chunk)
            user_ids = list(users_ids_mapping.keys())

            aggregated_data_list = await asyncio.gather(
                *(get_aggregated_tweets(client, user_id) for user_id in user_ids)
            )

            result.update({
                users_ids_mapping[user_id]: aggregated_data
                for user_id, aggregated_data in zip(user_ids, aggregated_data_list)
            })

    logger.info("aggregation_complete", {"users_count": len(result)})
    return result


if __name__ == "__main__":
    users = ["elonmusk", "naval", "sundarpichai"]
    result = asyncio.run(main(users))
    print(result)
