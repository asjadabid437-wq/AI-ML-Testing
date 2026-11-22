
# ============================================================
# PASSENGER SATISFACTION PREDICTION - STREAMLIT WEB APP
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("best_passenger_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Passenger Satisfaction AI",
    page_icon="✈️",
    layout="wide"
)

# Title
st.title("✈️ Passenger Satisfaction Prediction")

st.write(
    "Enter passenger details below to predict "
    "their airline satisfaction."
)

st.divider()

# Passenger Information Form

with st.form("passenger_form"):

    st.subheader("Passenger Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        customer_type = st.selectbox(
            "Customer Type",
            ["Loyal Customer", "disloyal Customer"]
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=25
        )

        travel_type = st.selectbox(
            "Type of Travel",
            ["Business travel", "Personal Travel"]
        )

        travel_class = st.selectbox(
            "Class",
            ["Business", "Eco", "Eco Plus"]
        )

        flight_distance = st.number_input(
            "Flight Distance",
            min_value=0,
            value=1000
        )

    with col2:

        wifi = st.slider("Inflight wifi service", 0, 5, 3)

        departure_time = st.slider(
            "Departure/Arrival time convenient", 0, 5, 3
        )

        online_booking = st.slider(
            "Ease of Online booking", 0, 5, 3
        )

        gate_location = st.slider("Gate location", 0, 5, 3)

        food = st.slider("Food and drink", 0, 5, 3)

        online_boarding = st.slider(
            "Online boarding", 0, 5, 3
        )

        seat_comfort = st.slider("Seat comfort", 0, 5, 3)

    with col3:

        entertainment = st.slider(
            "Inflight entertainment", 0, 5, 3
        )

        onboard_service = st.slider(
            "On-board service", 0, 5, 3
        )

        leg_room = st.slider("Leg room service", 0, 5, 3)

        baggage = st.slider("Baggage handling", 0, 5, 3)

        checkin = st.slider("Checkin service", 0, 5, 3)

        inflight_service = st.slider(
            "Inflight service", 0, 5, 3
        )

        cleanliness = st.slider("Cleanliness", 0, 5, 3)

    st.divider()

    st.subheader("Flight Delay Information")

    col4, col5 = st.columns(2)

    with col4:

        departure_delay = st.number_input(
            "Departure Delay in Minutes",
            min_value=0,
            value=0
        )

    with col5:

        arrival_delay = st.number_input(
            "Arrival Delay in Minutes",
            min_value=0,
            value=0
        )

    submitted = st.form_submit_button(
        "Predict Passenger Satisfaction",
        use_container_width=True
    )


# Prediction

if submitted:

    passenger = pd.DataFrame([{

        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": travel_type,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Inflight wifi service": wifi,
        "Departure/Arrival time convenient": departure_time,
        "Ease of Online booking": online_booking,
        "Gate location": gate_location,
        "Food and drink": food,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": entertainment,
        "On-board service": onboard_service,
        "Leg room service": leg_room,
        "Baggage handling": baggage,
        "Checkin service": checkin,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay

    }])

    prediction = model.predict(passenger)

    probabilities = model.predict_proba(passenger)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.success("Passenger is Predicted to be SATISFIED!")

    else:

        st.warning(
            "Passenger is Predicted to be NEUTRAL OR DISSATISFIED!"
        )

    col6, col7 = st.columns(2)

    with col6:

        st.metric(
            "Not Satisfied Probability",
            f"{probabilities[0] * 100:.2f}%"
        )

    with col7:

        st.metric(
            "Satisfied Probability",
            f"{probabilities[1] * 100:.2f}%"
        )

    st.write("Prediction Probability")

    probability_df = pd.DataFrame({
        "Satisfaction": [
            "Not Satisfied",
            "Satisfied"
        ],
        "Probability": probabilities
    })

    st.bar_chart(
        probability_df.set_index("Satisfaction")
    )

    st.caption(
        "This prediction is based on the trained Random Forest model."
    )
