#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Collect static files (for Django)
python manage.py collectstatic --noinput

# Apply migrations
python manage.py migrate
