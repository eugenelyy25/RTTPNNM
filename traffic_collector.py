name: Traffic & Weather Data Collection

on:
  schedule:
    # Runs every 30 minutes from 00:00 UTC (8am KL) to 13:30 UTC (9:30pm KL)
    - cron: '0,30 0-13 * * *'
  workflow_dispatch: # Allows manual trigger

jobs:
  collect-data:
    runs-on: ubuntu-latest
    timeout-minutes: 15 # Safety: Kills job if it freezes
    permissions:
      contents: write
      
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3

      # CRITICAL FIX: Frees ~4GB space to prevent "No space left on device" crash
      - name: Free Disk Space
        run: |
          sudo rm -rf /usr/share/dotnet
          sudo rm -rf /usr/local/lib/android
          sudo rm -rf /opt/ghc
          echo "Disk space freed."

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          
          # CRITICAL FIX: Install CPU-only PyTorch (saves 3GB space)
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
          
          # Install other requirements
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          
          # Install Ultralytics (uses existing CPU torch)
          pip install ultralytics
          
          # Pre-download YOLO model
          python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

      - name: Run Data Collection Script
        env:
          GOOGLE_MAPS_KEY: ${{ secrets.GOOGLE_MAPS_KEY }}
        # CRITICAL FIX: '-u' allows you to see logs in real-time
        run: python -u traffic_collector.py

      - name: Commit and Push Changes
        run: |
          git config --global user.name "GitHub Action"
          git config --global user.email "action@github.com"
          git add traffic_data.csv
          git commit -m "Auto-update traffic data [skip ci]" || echo "No changes to commit"
          git push
