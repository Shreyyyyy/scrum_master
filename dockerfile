# Use ubi9/ubi-minimal as the base image, which includes microdnf
FROM registry.access.redhat.com/ubi9/ubi-minimal

# Set working directory in the container
WORKDIR /app

# Install system dependencies required for Python, Pyppeteer, and Chromium
RUN microdnf install -y \
    python3.11 \
    python3.11-pip \
    gcc \
    postgresql-libs \
    atk \
    at-spi2-atk \
    cups-libs \
    dbus-libs \
    gdk-pixbuf2 \
    nspr \
    nss \
    libX11 \
    libXcomposite \
    libXdamage \
    libXrandr \
    mesa-libgbm \
    ca-certificates \
    tar \
    gzip \
    unzip \
    && microdnf clean all

# Install Chromium manually
RUN curl -L -o chromium.zip https://commondatastorage.googleapis.com/chromium-browser-snapshots/Linux_x64/1299788/chrome-linux.zip && \
    unzip chromium.zip -d /usr/lib/chromium-browser && \
    rm chromium.zip && \
    ln -s /usr/lib/chromium-browser/chrome-linux/chrome /usr/bin/chromium-browser

# Install pip for Python 3.11 and upgrade it
RUN python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --no-cache-dir --upgrade pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY .env .
COPY report_pdf_generator.py .

# Set environment variables for Pyppeteer and Chromium
ENV PYPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV PYPPETEER_ARGS="--no-sandbox"

# Command to run your application
CMD ["python3.11", "app.py"]
