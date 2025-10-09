import pytest
import sys
from pathlib import Path

repo_root = Path(__file__).parents[3]
sys.path.append(str(repo_root))

import module_factory
from GUI.Components.DropLabel import DropLabel


def test_compare_tle_readers(qtbot):
    # Arrange
    tle_file = repo_root / "TLEs" / "SatCount1.tle"
    assert tle_file.exists(), f"TLE file not found at {tle_file}"

    # --- Raw file ---
    print("\n=== RAW TLE FILE CONTENTS ===")
    with open(tle_file, "r") as f:
        for i, line in enumerate(f, 1):
            print(f"{i:02d}: {line.strip()}")

    # --- Reader 1: SatSimHandler ---
    sat_mod = module_factory.create_sat_sim_module()
    handler = sat_mod.handler

    print("\n=== HANDLER PARSED DATA ===")
    handler_data = handler.read_tle_file(str(tle_file))
    print(handler_data)

    # --- Reader 2: DropLabel ---
    label = DropLabel()
    print(label)
    qtbot.addWidget(label)

    parsed_from_label = {}

    def debug_slot(path):
        print(f"\n=== fileDropped SIGNAL EMITTED ===\nPath: {path}")
        # Log whatever DropLabel stores internally
        print("DropLabel parsed_data:")
        print(label.parsed_data)
        parsed_from_label = label.parsed_data
        # if hasattr(label, "parsed_data"):
        #     print("DropLabel parsed_data:")
        #     print(label.parsed_data)
        #     nonlocal parsed_from_label
        #     parsed_from_label = label.parsed_data

    label.fileDropped.connect(debug_slot)
    label.fileDropped.emit(str(tle_file))

    print("\n=== COMPARISON ===")
    print("Handler output:", handler_data)
    print("DropLabel output:", parsed_from_label)

    # Assert both produce something
    assert handler_data, "Handler produced no data"
    assert parsed_from_label, "DropLabel produced no data"
    
    # Assert both readers produce same output
    assert handler_data == parsed_from_label, "Parsed outputs differ between handler and DropLabel"