# Tweet Signal Task

A Python application for aggregating Twitter user data and tweet metrics.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tsyplakou/tweet-signal-aggregator
cd tweet-signal-task
```

## Running the Application

To run the application, execute:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
TWITTER_BEARER_TOKEN="YOUR_TOKEN" python app.py
```

## Running Tests

To run the tests, execute:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python -m pytest
```
