from typing import List, Dict

def generate_strategy(input_data, predicted_finish):

    strategy = []

    strategy.append(
        f"Predicted Finish Position: {predicted_finish:.2f}"
    )

    if predicted_finish <= 5:

        strategy.append(
            "Excellent chance of a Top-5 finish."
        )

    elif predicted_finish <= 10:

        strategy.append(
            "Expected Top-10 finish."
        )

    elif predicted_finish <= 20:

        strategy.append(
            "Competitive midfield performance expected."
        )

    else:

        strategy.append(
            "Outside Top-20 expected. Strategy improvements recommended."
        )

    return strategy

def analyze_start_position(start_position):

    if start_position <= 5:

        return "Excellent starting position."

    elif start_position <= 15:

        return "Good starting position."

    else:

        return "Poor starting position. Qualifying improvement recommended."
    
def analyze_driver_experience(experience):

    if experience >= 300:

        return "Highly experienced driver."

    elif experience >= 100:

        return "Experienced driver."

    else:

        return "Relatively inexperienced driver."
    
def analyze_driver_history(avg_finish):

    if avg_finish <= 10:

        return "Strong historical finishing performance."

    elif avg_finish <= 20:

        return "Average historical performance."

    else:

        return "Historical performance below average."
    
def analyze_team(avg_finish):

    if avg_finish <= 10:

        return "Team has strong historical results."

    elif avg_finish <= 20:

        return "Team performs consistently."

    else:

        return "Team performance needs improvement."
    
def analyze_track(track_type):

    if track_type == "Superspeedway":

        return "Superspeedway strategy: drafting and pit timing are critical."

    elif track_type == "Intermediate":

        return "Intermediate track: tire management is important."

    else:

        return "Short track: overtaking opportunities are higher."

def generate_strategy(input_data, predicted_finish):

    strategy = []

    strategy.append(
        f"🏁 Predicted Finish Position: {predicted_finish:.2f}"
    )

    if predicted_finish <= 5:

        strategy.append("Excellent chance of a Top-5 finish.")

    elif predicted_finish <= 10:

        strategy.append("Expected Top-10 finish.")

    elif predicted_finish <= 20:

        strategy.append("Competitive midfield performance expected.")

    else:

        strategy.append("Outside Top-20 expected.")

    strategy.append(
        analyze_start_position(
            input_data["Start"]
        )
    )

    strategy.append(
        analyze_driver_experience(
            input_data["Driver_Experience"]
        )
    )

    strategy.append(
        analyze_driver_history(
            input_data["Driver_Historical_Avg_Finish"]
        )
    )

    strategy.append(
        analyze_team(
            input_data["Team_Historical_Avg_Finish"]
        )
    )

    strategy.append(
        analyze_track(
            input_data["Track_Type"]
        )
    )

    return strategy

if __name__ == "__main__":

    sample = {

        "Start": 8,

        "Driver_Experience": 350,

        "Driver_Historical_Avg_Finish": 10.8,

        "Team_Historical_Avg_Finish": 11.3,

        "Track_Type": "Superspeedway"

    }

    strategy = generate_strategy(
        sample,
        10.32
    )

    print("\nRace Strategy\n")

    for item in strategy:

        print("•", item)

