

#-----------------------
# Constants

Seconds_Per_Day = 24 * 60 *60
Earth_Degrees = 360
Seconds_Per_Degree = Seconds_Per_Day / 360

# Convert seconds since midnight to degrees of rotation
def degrees_to_sec(deg: int):
    return int(round(deg * Seconds_Per_Degree)) % Seconds_Per_Day

def seconds_to_degrees(s: int):
    s = s % Seconds_Per_Day
    return int(round(s / Seconds_Per_Degree))