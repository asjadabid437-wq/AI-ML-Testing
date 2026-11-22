
# ============================================================
# PASSENGER SATISFACTION PREDICTION SYSTEM
# ============================================================

import joblib
import pandas as pd

# Load Trained Model
model = joblib.load("best_passenger_model.pkl")

print("\n======================================")
print(" PASSENGER SATISFACTION PREDICTION")
print("======================================")

# Passenger Information

gender = input("Enter Gender (Male/Female): ")

customer_type = input(
    "Enter Customer Type (Loyal Customer/disloyal Customer): "
)

age = int(input("Enter Age: "))

travel_type = input(
    "Enter Type of Travel (Business travel/Personal Travel): "
)

travel_class = input(
    "Enter Class (Business/Eco/Eco Plus): "
)

flight_distance = int(input("Enter Flight Distance: "))

wifi = int(input("Inflight wifi service (0-5): "))

departure_time = int(
    input("Departure/Arrival time convenient (0-5): ")
)

online_booking = int(
    input("Ease of Online booking (0-5): ")
)

gate_location = int(input("Gate location (0-5): "))

food = int(input("Food and drink (0-5): "))

online_boarding = int(input("Online boarding (0-5): "))

seat_comfort = int(input("Seat comfort (0-5): "))

entertainment = int(
    input("Inflight entertainment (0-5): ")
)

onboard_service = int(input("On-board service (0-5): "))

leg_room = int(input("Leg room service (0-5): "))

baggage = int(input("Baggage handling (0-5): "))

checkin = int(input("Checkin service (0-5): "))

inflight_service = int(input("Inflight service (0-5): "))

cleanliness = int(input("Cleanliness (0-5): "))

departure_delay = int(
    input("Departure Delay in Minutes: ")
)

arrival_delay = int(
    input("Arrival Delay in Minutes: ")
)


# Create Passenger DataFrame

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


# Make Prediction

prediction = model.predict(passenger)

probability = model.predict_proba(passenger)[0]

print("\n======================================")
print(" PREDICTION RESULT")
print("======================================")

if prediction[0] == 1:

    print("Passenger Satisfaction: SATISFIED")

else:

    print("Passenger Satisfaction: NEUTRAL OR DISSATISFIED")


print("\nPrediction Probabilities:")

print(
    "Not Satisfied:",
    round(probability[0] * 100, 2), "%"
)

print(
    "Satisfied:",
    round(probability[1] * 100, 2), "%"
)

print("\nPrediction Completed Successfully!")
