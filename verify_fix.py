"""
Verify that the improved fuzzy matching correctly resolves
garbled OCR output to known book titles.
"""
from ocr_utils import _correct_with_known_books

# Test cases: (garbled OCR input, expected corrected output)
test_cases = [
    ("Mathematicol Mothodt",  "Mathematical Methods"),
    ("Mather",                "Mathematical Methods"),
    ("Gi",                    "Gi"),  # Too short to match anything
    ("Enginoering Mathamatics", "Engineering Mathematics"),
    ("Phvsics",               "Physics"),
    ("Chemistri",             "Chemistry"),
    ("Comput3r Sci3nce",      "Computer Science"),
    ("Dato Structurcs",       "Data Structures"),
    ("Algorithnms",           "Algorithms"),
    ("Oporating Systoms",    "Operating Systems"),
    ("Databaso Systems",      "Database Systems"),
    ("Artificial Intellig",   "Artificial Intelligence"),
    ("Machina Learning",      "Machine Learning"),
    ("Doep Learning",         "Deep Learning"),
    ("Pyhton Programming",    "Python Programming"),
    ("Mathematical Methods",  "Mathematical Methods"),  # Exact
]

print("=" * 65)
print("  FUZZY MATCHING VERIFICATION")
print("=" * 65)

passed = 0
failed = 0

for raw, expected in test_cases:
    result = _correct_with_known_books(raw)
    status = "PASS" if result == expected else "FAIL"
    if status == "PASS":
        passed += 1
    else:
        failed += 1
    
    print(f"  [{status}] '{raw}' -> '{result}'")
    if status == "FAIL":
        print(f"         Expected: '{expected}'")

print("=" * 65)
print(f"  Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("=" * 65)
