from dataclasses import dataclass
import json

# Define data classes for options

@dataclass
class Location:
    lat: float
    long: float

@dataclass
class GroundStation:
    location: Location

@dataclass
class SatSimOptions:
    satellite_count: int
    constellation_type: str
    # Example of object inside an object... inside an object, reflecting JSON structure
    ground_station: GroundStation


def read_options_file():
    with open('test.json') as f:
        options = json.load(f)
        
    return options

def get_sat_sim_options(options):

    sat_sim_options = SatSimOptions(
        options["satellite_count"],
        options["constellation_type"],
        GroundStation(Location(**options["ground_station"]["location"]))
    )

    return sat_sim_options

def run_sat_sim(options):
    print(f'Running Satellite Simulator with these options: {options}')

if __name__ == "__main__":
    # Run helper function
    options = read_options_file()
   
    sat_sim_options = get_sat_sim_options(options["sat_sim"])
    run_sat_sim(sat_sim_options)