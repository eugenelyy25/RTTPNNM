import os
import cv2
import time
import requests
import numpy as np
import pandas as pd
import googlemaps
from datetime import datetime, timedelta
from ultralytics import YOLO

# ================= CONFIGURATION =================
# 1. API Keys (Set this in GitHub Secrets)
GMAPS_KEY = os.environ.get('GOOGLE_MAPS_KEY')
if not GMAPS_KEY:
    print("WARNING: No Google Maps API Key found. Traffic data will be empty.")

# 2. Camera Configuration (Mapped to your provided URLs)
CAMERA_MAP = {
    'NPE (E10) CAM 23 BRIDGE 19 KM12.8 WB': {
        'url': 'https://c2.fgies.com//sd-npe/NPE-23.jpg?',
        'ref': 'NPE-23-CAPTUREWITHINGREENLINE.jpg'
    },
    'SRT (E23) CAM 03 SEK17 KM2.6 EB': {
        'url': 'https://c12.fgies.com//sd-srt/SRT-03.jpg?',
        'ref': 'SRT-03-CAPTUREWITHINGREENLINE.jpg'
    },
    'SRT (E23) CAM 06 KIARA KM5.65 MED': {
        'url': 'https://c12.fgies.com//sd-srt/SRT-06.jpg?',
        'ref': 'SRT-06-CAPTUREWITHINGREENLINE.jpg'
    },
    'SPE (E39) CAM 02 SEPUTIH KM2.2 NB': {
        'url': 'https://c12.fgies.com//sd-spe/SPE-02.jpg?',
        'ref': 'SPE-02-CAPTUREWITHINGREENLINE.jpg'
    },
    'SPE (E39) CAM 03 BUKIT DESA KM3.1 NB': {
        'url': 'https://c12.fgies.com//sd-spe/SPE-03.jpg?',
        'ref': 'SPE-03-CAPTUREWITHINGREENLINE.jpg'
    }
}

# 3. Fixed Routes (Extracted from your Metadata_Example)
ROUTES = [
    {'origin_location': 'GATE 1 (KL GATE)', 'origin_coordinate': '3.118774, 101.663074', 'destination_location': 'NPE (E10) CAM 14 PBAHARU KM13.3 EB', 'destination_coordinate': '3.116669, 101.671921', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 1, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'NPE (E10) CAM 23 BRIDGE 19 KM12.8 WB', 'origin_coordinate': '3.113084, 101.673034', 'destination_location': 'GATE 1 (KL GATE)', 'destination_coordinate': '3.118774, 101.663074', 'direction': 'ENTERING', 'EG1': 1, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 1},
    {'origin_location': 'NPE (E10) CAM 24 VMS BANGSAR KM14 WB', 'origin_coordinate': '3.121334, 101.674702', 'destination_location': 'GATE 1 (KL GATE)', 'destination_coordinate': '3.118774, 101.663074', 'direction': 'ENTERING', 'EG1': 1, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'SRT (E23) CAM 03 SEK17 KM2.6 EB', 'origin_coordinate': '3.132604, 101.635048', 'destination_location': 'GATE 2 (PJ GATE)', 'destination_coordinate': '3.115642, 101.650832', 'direction': 'ENTERING', 'EG1': 0, 'LG1': 0, 'EG2': 1, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 1},
    {'origin_location': 'SRT (E23) CAM 06 KIARA KM5.65 MED', 'origin_coordinate': '3.136340, 101.656388', 'destination_location': 'GATE 4 (DAMANSARA GATE)', 'destination_coordinate': '3.137486, 101.658259', 'direction': 'ENTERING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 1, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 1},
    {'origin_location': 'SPE (E39) CAM 01 KERINCHI KM0.2 NB', 'origin_coordinate': '3.114883, 101.665004', 'destination_location': 'GATE 1 (KL GATE)', 'destination_coordinate': '3.118774, 101.663074', 'direction': 'ENTERING', 'EG1': 1, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'SPE (E39) CAM 02 SEPUTIH KM2.2 NB', 'origin_coordinate': '3.111576, 101.687789', 'destination_location': 'GATE 1 (KL GATE)', 'destination_coordinate': '3.118774, 101.663074', 'direction': 'ENTERING', 'EG1': 1, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 1},
    {'origin_location': 'GATE 1 (KL GATE)', 'origin_coordinate': '3.118774, 101.663074', 'destination_location': 'SPE (E39) CAM 03 BUKIT DESA KM3.1 NB', 'destination_coordinate': '3.111892, 101.687890', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 1, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 1},
    {'origin_location': 'FSKTM', 'origin_coordinate': '3.127503, 101.650681', 'destination_location': 'GATE 1 (KL GATE)', 'destination_coordinate': '3.118774, 101.663074', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 1, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'FSKTM', 'origin_coordinate': '3.127503, 101.650681', 'destination_location': 'GATE 2 (PJ GATE)', 'destination_coordinate': '3.115642, 101.650832', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 1, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'FSKTM', 'origin_coordinate': '3.127503, 101.650681', 'destination_location': 'GATE 3 (MAHSA GATE)', 'destination_coordinate': '3.118839, 101.651122', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 1, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'FSKTM', 'origin_coordinate': '3.127503, 101.650681', 'destination_location': 'GATE 4 (DAMANSARA GATE)', 'destination_coordinate': '3.137486, 101.658259', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 1, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 1 (KL GATE)', 'origin_coordinate': '3.118774, 101.663074', 'destination_location': 'FSKTM', 'destination_coordinate': '3.127503, 101.650681', 'direction': 'ENTERING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 1, 'active_CAM': 0},
    {'origin_location': 'GATE 1 (KL GATE)', 'origin_coordinate': '3.118774, 101.663074', 'destination_location': 'GATE 2 (PJ GATE)', 'destination_coordinate': '3.115642, 101.650832', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 1, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 1 (KL GATE)', 'origin_coordinate': '3.118774, 101.663074', 'destination_location': 'GATE 3 (MAHSA GATE)', 'destination_coordinate': '3.118839, 101.651122', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 1, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 1 (KL GATE)', 'origin_coordinate': '3.118774, 101.663074', 'destination_location': 'GATE 4 (DAMANSARA GATE)', 'destination_coordinate': '3.137486, 101.658259', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 1, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 2 (PJ GATE)', 'origin_coordinate': '3.115642, 101.650832', 'destination_location': 'FSKTM', 'destination_coordinate': '3.127503, 101.650681', 'direction': 'ENTERING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 1, 'active_CAM': 0},
    {'origin_location': 'GATE 2 (PJ GATE)', 'origin_coordinate': '3.115642, 101.650832', 'destination_location': 'GATE 1 (KL GATE)', 'destination_coordinate': '3.118774, 101.663074', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 1, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 2 (PJ GATE)', 'origin_coordinate': '3.115642, 101.650832', 'destination_location': 'GATE 3 (MAHSA GATE)', 'destination_coordinate': '3.118839, 101.651122', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 1, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 2 (PJ GATE)', 'origin_coordinate': '3.115642, 101.650832', 'destination_location': 'GATE 4 (DAMANSARA GATE)', 'destination_coordinate': '3.137486, 101.658259', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 1, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 3 (MAHSA GATE)', 'origin_coordinate': '3.118839, 101.651122', 'destination_location': 'FSKTM', 'destination_coordinate': '3.127503, 101.650681', 'direction': 'ENTERING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 1, 'active_CAM': 0},
    {'origin_location': 'GATE 3 (MAHSA GATE)', 'origin_coordinate': '3.118839, 101.651122', 'destination_location': 'GATE 2 (PJ GATE)', 'destination_coordinate': '3.115642, 101.650832', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 1, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 3 (MAHSA GATE)', 'origin_coordinate': '3.118839, 101.651122', 'destination_location': 'GATE 1 (KL GATE)', 'destination_coordinate': '3.118774, 101.663074', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 1, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 3 (MAHSA GATE)', 'origin_coordinate': '3.118839, 101.651122', 'destination_location': 'GATE 4 (DAMANSARA GATE)', 'destination_coordinate': '3.137486, 101.658259', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 1, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 4 (DAMANSARA GATE)', 'origin_coordinate': '3.137486, 101.658259', 'destination_location': 'FSKTM', 'destination_coordinate': '3.127503, 101.650681', 'direction': 'ENTERING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 1, 'active_CAM': 0},
    {'origin_location': 'GATE 4 (DAMANSARA GATE)', 'origin_coordinate': '3.137486, 101.658259', 'destination_location': 'GATE 2 (PJ GATE)', 'destination_coordinate': '3.115642, 101.650832', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 1, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 4 (DAMANSARA GATE)', 'origin_coordinate': '3.137486, 101.658259', 'destination_location': 'GATE 3 (MAHSA GATE)', 'destination_coordinate': '3.118839, 101.651122', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 0, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 1, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0},
    {'origin_location': 'GATE 4 (DAMANSARA GATE)', 'origin_coordinate': '3.137486, 101.658259', 'destination_location': 'GATE 1 (KL GATE)', 'destination_coordinate': '3.118774, 101.663074', 'direction': 'LEAVING', 'EG1': 0, 'LG1': 1, 'EG2': 0, 'LG2': 0, 'EG3': 0, 'LG3': 0, 'EG4': 0, 'LG4': 0, 'FSKTM': 0, 'active_CAM': 0}
]

# ================= HELPER FUNCTIONS =================

def check_holidays(current_date):
    """Parses HList_KL.txt to check holiday status."""
    is_holiday = 0
    is_school_holiday = 0
    is_festive = 0
    
    date_str = current_date.strftime("%d.%m.%y")
    
    try:
        with open('HList_KL.txt', 'r') as f:
            lines = f.readlines()
            
        mode = None # "PUBLIC" or "SCH"
        for line in lines:
            line = line.strip()
            if "PUBLIC HOLIDAY" in line:
                mode = "PUBLIC"
                continue
            elif "SCHOOL HOLIDAY" in line or line.startswith("SCH"):
                mode = "SCH"
                continue
                
            if mode == "PUBLIC":
                if date_str in line:
                    is_holiday = 1
                    if "FESTIVE" in line.upper():
                        is_festive = 1
            elif mode == "SCH":
                # School holiday lines often look like dates or ranges
                if date_str in line:
                    is_school_holiday = 1
    except Exception as e:
        print(f"Error reading holiday file: {e}")
        
    return is_holiday, is_school_holiday, is_festive

def get_weather(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m,visibility",
            "timezone": "Asia/Singapore" # KL Time
        }
        resp = requests.get(url, params=params)
        data = resp.json()['current']
        
        # Rain: Codes 51-67, 80-82, 95-99
        rain_codes = [51,53,55,61,63,65,66,67,80,81,82,95,96,99]
        is_raining = 1 if data['weather_code'] in rain_codes else 0
        
        # Visibility: OpenMeteo gives meters. >10km is Clear.
        is_clear = 1 if data['visibility'] >= 10000 else 0
        
        # Windy: > 20 km/h
        is_windy = 1 if data['wind_speed_10m'] > 20 else 0
        
        # Hot: >= 30 C
        is_hot = 1 if data['temperature_2m'] >= 30 else 0
        
        return is_raining, is_clear, is_windy, is_hot
    except Exception as e:
        print(f"Weather API Error: {e}")
        return 0, 1, 0, 0 # Defaults

def get_traffic_google(origin, dest):
    if not GMAPS_KEY: return 0, 0
    
    try:
        gmaps = googlemaps.Client(key=GMAPS_KEY)
        # Using directions to get duration in traffic
        now = datetime.now()
        directions = gmaps.directions(origin, dest, mode="driving", departure_time=now)
        
        if directions:
            leg = directions[0]['legs'][0]
            duration_min = leg['duration']['value'] / 60
            duration_traffic_min = leg.get('duration_in_traffic', leg['duration'])['value'] / 60
            
            # Jam Logic: If traffic makes it 25% slower than standard
            is_jam = 1 if duration_traffic_min > (duration_min * 1.25) else 0
            
            return int(duration_traffic_min), is_jam
    except Exception as e:
        print(f"Google Maps Error: {e}")
    return 0, 0

# ================= YOLO TRAFFIC ANALYSIS =================
model = YOLO('yolo11n.pt') 

def analyze_camera_traffic(camera_name):
    if camera_name not in CAMERA_MAP:
        return "NA"
    
    conf = CAMERA_MAP[camera_name]
    
    # 1. Get Reference Mask
    if not os.path.exists(conf['ref']):
        print(f"Missing reference image for {camera_name}")
        return "NA"
    
    # Generate Mask from Green Lines
    ref_img = cv2.imread(conf['ref'])
    hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
    
    # Fill Mask (Row-wise fill between lines)
    filled_mask = np.zeros_like(mask)
    for r in range(mask.shape[0]):
        indices = np.where(mask[r, :] > 0)[0]
        if len(indices) >= 2:
            filled_mask[r, indices[0]:indices[-1]] = 255
            
    # 2. Fetch Live Image
    try:
        resp = requests.get(conf['url'], timeout=10)
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        frame = cv2.imdecode(arr, -1)
    except:
        return "NA"
    
    # 3. YOLO Detection
    results = model(frame, verbose=False, conf=0.25) # conf=0.25 standard
    
    # 4. Calculate Density
    total_zone_pixels = np.count_nonzero(filled_mask)
    if total_zone_pixels == 0: return "NA"
    
    vehicle_area_pixels = 0
    for box in results[0].boxes:
        # Check if center is in zone
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        
        if 0 <= cy < filled_mask.shape[0] and 0 <= cx < filled_mask.shape[1]:
            if filled_mask[cy, cx] == 255:
                # Add area of box to total vehicle area (clipping to image size not strictly necessary for estimation)
                vehicle_area_pixels += (x2-x1) * (y2-y1)

    density = (vehicle_area_pixels / total_zone_pixels) * 100
    
    if density >= 60: return "HEAVY"
    elif density >= 40: return "MODERATE"
    else: return "LIGHT"

# ================= MAIN EXECUTION =================

def main():
    # 1. Time Setup (KL Time)
    utc_now = datetime.utcnow()
    kl_now = utc_now + timedelta(hours=8)
    
    # Ensure between 0800 and 2100
    current_hour = kl_now.hour
    if not (8 <= current_hour <= 21):
        print(f"Current time {kl_now} is outside operating hours (0800-2100). Exiting.")
        return

    # Round to nearest 30 mins for timestamp label
    minute = kl_now.minute
    if minute < 15:
        ts_min = "00"
    elif minute < 45:
        ts_min = "30"
    else:
        ts_min = "00"
        # If rounding up to next hour, careful, but simplified here:
        # We just want the label to strictly match 0800, 0830 format requested
        pass 
    
    # Exact formatted timestamp requested (e.g. 830, 1400)
    # Logic: if minute >= 30, use 30. else 00.
    display_min = "30" if minute >= 30 else "00"
    timestamp = f"{current_hour}{display_min}"
    # Remove leading zero if needed? Example shows '830' not '0830'.
    # If 0800 -> 800. Let's strictly follow example '830'.
    timestamp = str(int(timestamp)) 

    # 2. Date/Day Vars
    date_str = kl_now.strftime("%d.%m.%y")
    day_name = kl_now.strftime("%A").upper()
    month_name = kl_now.strftime("%B").upper()
    is_weekend = "WEEKEND" if kl_now.weekday() >= 5 else "WEEKDAY"
    
    # Time of Day
    if 8 <= current_hour < 17:
        time_of_day = "DAY"
    elif 17 <= current_hour < 17.5: # 1700-1730 gap? treated as DAY
        time_of_day = "DAY"
    else:
        time_of_day = "NIGHT" # 1730 - 2100
        
    # Peak Hour
    # 0800-0900, 1200-1300, 1730-1900
    is_peak = 0
    t_val = current_hour * 100 + int(display_min)
    if (800 <= t_val <= 900) or (1200 <= t_val <= 1300) or (1730 <= t_val <= 1900):
        is_peak = 1
        
    # Holiday Check
    is_hol, is_schol, is_fest = check_holidays(kl_now)

    # 3. Iterate Routes
    rows = []
    
    # Cache YOLO results to avoid duplicate processing if multiple routes use same camera
    camera_cache = {} 
    
    for route in ROUTES:
        row = route.copy() # Start with fixed columns
        
        # Add Time/Date Columns
        row['timestamp'] = timestamp
        row['is_PeakHour'] = is_peak
        row['time_of_day'] = time_of_day
        row['day_name'] = day_name
        row['type_of_day'] = is_weekend
        row['month'] = month_name
        row['date'] = date_str
        
        # Add Holiday Columns
        row['is_Holiday'] = is_hol
        row['is_SchoolHoliday'] = is_schol
        row['is_Festive'] = is_fest
        
        # Weather (Origin)
        lat, lon = map(float, row['origin_coordinate'].split(','))
        raining, clear, windy, hot = get_weather(lat, lon)
        row['origin_Raining'] = raining
        row['origin _ClearVisibility'] = clear
        row['origin_Windy'] = windy
        row['origin_Hot'] = hot
        
        # Traffic (Google)
        duration, is_jam = get_traffic_google(row['origin_coordinate'], row['destination_coordinate'])
        row['calculated_journeytime_minute'] = int(duration)
        row['is_Jam'] = is_jam
        
        # YOLO Analysis
        conc = "NA"
        if row['active_CAM'] == 1:
            # Determine which camera. Assuming Origin Name matches Camera Map keys
            cam_name = row['origin_location']
            
            # If origin isn't a camera, check destination (rare case in your list)
            if cam_name not in CAMERA_MAP and row['destination_location'] in CAMERA_MAP:
                cam_name = row['destination_location']
            
            if cam_name in CAMERA_MAP:
                if cam_name in camera_cache:
                    conc = camera_cache[cam_name]
                else:
                    conc = analyze_camera_traffic(cam_name)
                    camera_cache[cam_name] = conc
        
        row['origin_VehicleConcentration'] = conc
        
        # Data Pass: 1 if everything essential is there
        # We consider 'Pass' if journey time > 0 (API worked)
        row['data_Pass'] = 1 if row['calculated_journeytime_minute'] > 0 else 0
        
        rows.append(row)
        print(f"Processed: {row['origin_location']} -> {row['destination_location']}")
        time.sleep(0.5) # Slight delay to be nice to APIs

    # 4. Save to CSV
    df = pd.DataFrame(rows)
    
    # Reorder columns to match Example strictly
    cols = ['timestamp', 'is_PeakHour', 'time_of_day', 'day_name', 'type_of_day', 'month', 'date', 
            'origin_location', 'origin_coordinate', 'destination_location', 'destination_coordinate', 
            'direction', 'EG1', 'LG1', 'EG2', 'LG2', 'EG3', 'LG3', 'EG4', 'LG4', 'FSKTM', 
            'active_CAM', 'origin_Raining', 'origin _ClearVisibility', 'origin_Windy', 'origin_Hot', 
            'origin_VehicleConcentration', 'is_Holiday', 'is_SchoolHoliday', 'is_Festive', 
            'calculated_journeytime_minute', 'data_Pass', 'is_Jam']
    
    df = df[cols]
    
    file_name = 'traffic_data.csv'
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        df.to_csv(file_name, index=False)
        
    print("Data collection complete.")

if __name__ == "__main__":
    main()