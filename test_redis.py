import redis
import os
from urllib.parse import urlparse

REDIS_URL = "env.REDIS_URL"

try:
    # Parse the URL to verify it
    parsed = urlparse(REDIS_URL)
    print(f"Parsed URL: Scheme={parsed.scheme}, Hostname={parsed.hostname}, Port={parsed.port}")
    
    # Connect using strict Redis constructor
    r = redis.Redis(
        host=parsed.hostname,
        port=parsed.port,
        password=parsed.password,
        ssl=False,  # Disable SSL for now
        decode_responses=True
    )
    
    print("Redis ping response:", r.ping())  # Should return True
except Exception as e:
    print("Connection failed:", str(e))