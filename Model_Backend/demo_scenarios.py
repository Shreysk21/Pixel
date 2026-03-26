DEMO_SCENARIOS = {
    "Scenario 1 — Clear shipment": {
        "image": "test_images/test_clear.jpg",
        "declared": ["Electronic components", "Circuit boards", "Cables"],
        "expected_result": "CLEAR",
    },
    "Scenario 2 — Undeclared item": {
        "image": "test_images/test_suspicious.jpg",
        "declared": ["Clothing", "Shoes"],
        "expected_result": "SUSPICIOUS — undeclared metal object",
    },
    "Scenario 3 — Prohibited item": {
        "image": "test_images/test_prohibited.jpg",
        "declared": ["Books", "Documents"],
        "expected_result": "PROHIBITED — firearm detected",
    },
}
