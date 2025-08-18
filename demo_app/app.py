import streamlit as st
import requests
import json
from pathlib import Path

API_URL = st.secrets.get("API_URL", "http://api:8000/predict")  # docker compose service name
LOCAL_API_URL = "http://127.0.0.1:8000/predict"  # for local dev without compose

st.set_page_config(page_title="Airbnb Price Demo", page_icon="🏠", layout="centered")

st.title("Airbnb Price Prediction")
st.caption("Powered by FastAPI and your trained ensemble model")

# Default payload
default_payload = {
    "accommodates": 2,
    "bathrooms": 1.0,
    "bedrooms": 1.0,
    "beds": 1.0,
    "minimum_nights": 2,
    "maximum_nights": 1125,
    "number_of_reviews": 35,
    "review_scores_rating": 92.0,
    "host_age_days": 2000,
    "days_since_last_review": 100,
    "avg_review_score": 90.0,
    "season": 1,
    "distance_to_center_km": 2.5,
    "Has_Wifi": 1,
    "Has_Kitchen": 1,
    "Has_Heating": 1,
    "Has_TV": 1,
    "Has_Essentials": 1,
    "Has_Hair_Dryer": 0,
    "Has_Iron": 1,
    "Has_Free_Parking": 0,
    "Has_Hangers": 1,
    "Has_Laptop_Friendly_Workspace": 1,
    "host_has_profile_pic_binary": 1,
    "host_identity_verified_binary": 1,
    "host_is_superhost": 1,
    "instant_bookable": 1,
    "neighbourhood_cleansed": "Downtown",
    "room_type": "Entire home/apt"
}

st.subheader("Input")
with st.form("predict_form", clear_on_submit=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        accommodates = st.number_input("accommodates", 1, 20, default_payload["accommodates"])
        bathrooms = st.number_input("bathrooms", 0.5, 10.0, default_payload["bathrooms"], step=0.5)
        bedrooms = st.number_input("bedrooms", 0.0, 20.0, default_payload["bedrooms"], step=1.0)
        beds = st.number_input("beds", 0.0, 20.0, default_payload["beds"], step=1.0)
        minimum_nights = st.number_input("minimum_nights", 1, 1000, default_payload["minimum_nights"])
        maximum_nights = st.number_input("maximum_nights", 1, 5000, default_payload["maximum_nights"])
        distance_to_center_km = st.number_input("distance_to_center_km", 0.0, 50.0, default_payload["distance_to_center_km"])

    with c2:
        number_of_reviews = st.number_input("number_of_reviews", 0, 1000, default_payload["number_of_reviews"])
        review_scores_rating = st.number_input("review_scores_rating", 0.0, 100.0, default_payload["review_scores_rating"])
        host_age_days = st.number_input("host_age_days", 0, 10000, default_payload["host_age_days"])
        days_since_last_review = st.number_input("days_since_last_review", 0, 5000, default_payload["days_since_last_review"])
        avg_review_score = st.number_input("avg_review_score", 0.0, 100.0, default_payload["avg_review_score"])
        season = st.selectbox("season, 0 winter, 1 spring, 2 summer, 3 fall", [0, 1, 2, 3], index=1)

    with c3:
        room_type = st.selectbox("room_type", ["Entire home/apt", "Private room", "Shared room"], index=0)
        neighbourhood_cleansed = st.text_input("neighbourhood_cleansed", default_payload["neighbourhood_cleansed"])
        Has_Wifi = st.checkbox("Has_Wifi", value=True)
        Has_Kitchen = st.checkbox("Has_Kitchen", value=True)
        Has_Heating = st.checkbox("Has_Heating", value=True)
        Has_TV = st.checkbox("Has_TV", value=True)
        Has_Essentials = st.checkbox("Has_Essentials", value=True)
        Has_Hair_Dryer = st.checkbox("Has_Hair_Dryer", value=False)
        Has_Iron = st.checkbox("Has_Iron", value=True)
        Has_Free_Parking = st.checkbox("Has_Free_Parking", value=False)
        Has_Hangers = st.checkbox("Has_Hangers", value=True)
        Has_Laptop_Friendly_Workspace = st.checkbox("Has_Laptop_Friendly_Workspace", value=True)
        host_has_profile_pic_binary = st.checkbox("host_has_profile_pic_binary", value=True)
        host_identity_verified_binary = st.checkbox("host_identity_verified_binary", value=True)
        host_is_superhost = st.checkbox("host_is_superhost", value=True)
        instant_bookable = st.checkbox("instant_bookable", value=True)

    submitted = st.form_submit_button("Predict")

payload = {
    "accommodates": int(accommodates) if "accommodates" in locals() else default_payload["accommodates"],
    "bathrooms": float(bathrooms) if "bathrooms" in locals() else default_payload["bathrooms"],
    "bedrooms": float(bedrooms) if "bedrooms" in locals() else default_payload["bedrooms"],
    "beds": float(beds) if "beds" in locals() else default_payload["beds"],
    "minimum_nights": int(minimum_nights) if "minimum_nights" in locals() else default_payload["minimum_nights"],
    "maximum_nights": int(maximum_nights) if "maximum_nights" in locals() else default_payload["maximum_nights"],
    "number_of_reviews": int(number_of_reviews) if "number_of_reviews" in locals() else default_payload["number_of_reviews"],
    "review_scores_rating": float(review_scores_rating) if "review_scores_rating" in locals() else default_payload["review_scores_rating"],
    "host_age_days": int(host_age_days) if "host_age_days" in locals() else default_payload["host_age_days"],
    "days_since_last_review": int(days_since_last_review) if "days_since_last_review" in locals() else default_payload["days_since_last_review"],
    "avg_review_score": float(avg_review_score) if "avg_review_score" in locals() else default_payload["avg_review_score"],
    "season": int(season) if "season" in locals() else default_payload["season"],
    "distance_to_center_km": float(distance_to_center_km) if "distance_to_center_km" in locals() else default_payload["distance_to_center_km"],
    "Has_Wifi": int(Has_Wifi) if "Has_Wifi" in locals() else default_payload["Has_Wifi"],
    "Has_Kitchen": int(Has_Kitchen) if "Has_Kitchen" in locals() else default_payload["Has_Kitchen"],
    "Has_Heating": int(Has_Heating) if "Has_Heating" in locals() else default_payload["Has_Heating"],
    "Has_TV": int(Has_TV) if "Has_TV" in locals() else default_payload["Has_TV"],
    "Has_Essentials": int(Has_Essentials) if "Has_Essentials" in locals() else default_payload["Has_Essentials"],
    "Has_Hair_Dryer": int(Has_Hair_Dryer) if "Has_Hair_Dryer" in locals() else default_payload["Has_Hair_Dryer"],
    "Has_Iron": int(Has_Iron) if "Has_Iron" in locals() else default_payload["Has_Iron"],
    "Has_Free_Parking": int(Has_Free_Parking) if "Has_Free_Parking" in locals() else default_payload["Has_Free_Parking"],
    "Has_Hangers": int(Has_Hangers) if "Has_Hangers" in locals() else default_payload["Has_Hangers"],
    "Has_Laptop_Friendly_Workspace": int(Has_Laptop_Friendly_Workspace) if "Has_Laptop_Friendly_Workspace" in locals() else default_payload["Has_Laptop_Friendly_Workspace"],
    "host_has_profile_pic_binary": int(host_has_profile_pic_binary) if "host_has_profile_pic_binary" in locals() else default_payload["host_has_profile_pic_binary"],
    "host_identity_verified_binary": int(host_identity_verified_binary) if "host_identity_verified_binary" in locals() else default_payload["host_identity_verified_binary"],
    "host_is_superhost": int(host_is_superhost) if "host_is_superhost" in locals() else default_payload["host_is_superhost"],
    "instant_bookable": int(instant_bookable) if "instant_bookable" in locals() else default_payload["instant_bookable"],
    "neighbourhood_cleansed": neighbourhood_cleansed if "neighbourhood_cleansed" in locals() else default_payload["neighbourhood_cleansed"],
    "room_type": room_type if "room_type" in locals() else default_payload["room_type"]
}

st.write("Request preview:")
st.code(json.dumps(payload, indent=2))

if submitted:
    try:
        # Try docker URL first, then local URL
        try:
            r = requests.post(API_URL, json=payload, timeout=10)
        except Exception:
            r = requests.post(LOCAL_API_URL, json=payload, timeout=10)

        if r.status_code == 200:
            res = r.json()
            st.success(f"Predicted nightly price: {res.get('predicted_price')} {res.get('currency', 'CAD')}")
            st.caption(f"request_id: {res.get('request_id', 'n/a')}, latency: {res.get('latency_ms', 'n/a')} ms")
        else:
            st.error(f"Error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
