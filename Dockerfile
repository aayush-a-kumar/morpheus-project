FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for scientific Python packages (compilers, git, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and wheel
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy dependency configuration files first 
# (This leverages Docker's caching so it doesn't reinstall everything if code changes)
COPY requirements.txt* pyproject.toml* setup.py* ./

# Install Python dependencies if requirements.txt exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy the rest of simulation code into the container
COPY . .

# If your repo is set up as an installable package, install it in editable mode
RUN pip install --no-cache-dir -e . || true

# Set the default command (change 'main.py' to whatever your main script/entry point is)
CMD ["python", "main.py"]