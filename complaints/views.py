from django.shortcuts import render
from .models import Complaint
from .ml import predict_category, predict_priority, detect_duplicate

def home(request):
    result = None

    if request.method == "POST":
        student_name = request.POST.get("student_name")
        description = request.POST.get("description")

        if student_name and description:
            category = predict_category(description)
            priority = predict_priority(description)
            duplicate = detect_duplicate(description)

            departments = {
                "IT & Wi-Fi": "IT Department",
                "Hostel": "Hostel Management",
                "Academic": "Academic Office",
                "Infrastructure": "Maintenance",
                "Transport": "Transport Office"
            }

            Complaint.objects.create(
                student_name=student_name,
                description=description,
                category=category,
                priority=priority,
                department=departments.get(category, "Administration"),
                is_duplicate=duplicate
            )

            result = {
                "category": category,
                "priority": priority,
                "department": departments.get(category, "Administration"),
                "duplicate": duplicate
            }

    return render(request, "complaints/home.html", {"result": result})