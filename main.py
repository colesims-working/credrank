import os

import uvicorn

# Dev entry point: default to auth-disabled so local runs "just work".
# Production uses uvicorn directly (see Dockerfile) and must set
# BASIC_AUTH_PASSWORD explicitly.
os.environ.setdefault("ALLOW_UNAUTH", "1")


def main():
    uvicorn.run("credrank.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
