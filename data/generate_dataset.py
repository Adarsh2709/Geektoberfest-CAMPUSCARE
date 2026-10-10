import csv
import random

random.seed(42)

templates = {
    "IT & Wi-Fi": [
        "Wi-Fi is not working in {}",
        "Internet connection is very slow in {}",
        "Computer lab network is down in {}",
        "Unable to connect to campus Wi-Fi near {}",
        "The internet keeps disconnecting in {}",
        "Wi-Fi password is not working in {}",
        "Network access is unavailable in {}",
        "Campus internet has stopped working in {}"
    ],

    "Hostel": [
        "No water supply in {} hostel",
        "Fan is not working in {} hostel room",
        "Hostel bathroom needs urgent cleaning",
        "The water cooler is not working in {} hostel",
        "There is a power problem in {} hostel",
        "Hostel room light is not working",
        "The hostel room tap is leaking",
        "Students are facing maintenance problems in {} hostel"
    ],

    "Academic": [
        "My attendance is incorrect in {}",
        "Internal marks are missing for {}",
        "The exam schedule has a problem",
        "Unable to access the course material for {}",
        "Teacher has not updated attendance",
        "My marks are not visible on the portal",
        "There is an issue with my timetable",
        "I cannot register for {} course"
    ],

    "Infrastructure": [
        "Classroom projector is not working in {}",
        "Lights are not working in {}",
        "Broken desks need repair in {}",
        "The classroom fan is damaged",
        "There is a maintenance issue in {} building",
        "The staircase railing is damaged",
        "The laboratory equipment needs repair",
        "The classroom door is broken"
    ],

    "Transport": [
        "College bus is arriving late",
        "The bus route for {} is not available",
        "College bus is overcrowded",
        "Bus driver is skipping the {} stop",
        "The transport schedule is incorrect",
        "Bus service is unavailable today",
        "The bus has frequent delays",
        "There is no bus during the evening"
    ]
}

places = [
    "Block A", "Block B", "Block C", "Block D",
    "Central Library", "Main Building", "Science Building",
    "Boys Hostel", "Girls Hostel", "Computer Lab",
    "Seminar Hall", "Workshop"
]

high_words = [
    "Urgent", "Emergency", "Fire", "Accident",
    "Unsafe", "Medical", "Security", "Immediately"
]

rows = []

for category, category_templates in templates.items():

    for _ in range(80):

        sentence = random.choice(category_templates)
        complaint = sentence.format(random.choice(places))

        priority = random.choices(
            ["Low", "Medium", "High"],
            weights=[20, 60, 20]
        )[0]

        if priority == "High":
            complaint = random.choice(high_words) + ": " + complaint

        rows.append({
            "complaint": complaint,
            "category": category,
            "priority": priority
        })

random.shuffle(rows)

with open("data/complaints.csv", "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["complaint", "category", "priority"]
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"Created {len(rows)} complaints.")