
### Technical Task: Tweet Signal Aggregator

**Context**  
We are building a system that captures narrative signals from Twitter. For this test, you will build a small async backend script that fetches and aggregates tweet data based on a list of Twitter accounts.

---

### Objective

Build a script in Python that:

1. Accepts a list of Twitter usernames as input
2. Asynchronously fetches their latest tweets (max 10 per user)
3. Aggregates:
   - Total tweets per user
   - Total likes and retweets per user
   - Timestamp of the most recent tweet
4. Returns a simple JSON summary

---

### Requirements

- Language: Python 3.9+
- Async fetching (e.g., using `aiohttp`, `httpx`, etc.)
- Output: JSON with per-user metrics
- Use either:
  - Twitter API (v2, if you have access)
  - OR scrape tweet data via public endpoints (your choice)

---

### Input Example

```python
usernames = ["sundarpichai", "naval", "elonmusk"]
```

---

### Output Example

```json
{
  "elonmusk": {
    "total_tweets": 10,
    "total_likes": 21890,
    "total_retweets": 5321,
    "latest_tweet": "2025-09-30T19:22:00Z"
  },
  "naval": {
    "total_tweets": 7,
    "total_likes": 8123,
    "total_retweets": 203,
    "latest_tweet": "2025-09-29T17:05:00Z"
  }
}
```

---

### Notes

- You are free to use any method to extract tweets.
- Clean, readable code matters more than perfection.
- Feel free to use mock data if you don’t have Twitter API access.

---

### Time Expectation

The task is designed to be completed in **2–3 hours max**.  
It reflects real patterns of the work we are doing.
