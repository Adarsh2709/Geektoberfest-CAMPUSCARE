from django.test import TestCase

from .models import Complaint
from .ml import predict_category, predict_priority


class ComplaintModelTest(TestCase):

    def test_complaint_creation(self):
        complaint = Complaint.objects.create(
            student_name="Test Student",
            description="Wi-Fi is not working in Block A",
            category="IT & Wi-Fi",
            priority="Medium",
            department="IT Department"
        )

        self.assertEqual(complaint.student_name, "Test Student")
        self.assertEqual(complaint.category, "IT & Wi-Fi")
        self.assertEqual(complaint.priority, "Medium")


class ComplaintMLTest(TestCase):

    def test_category_prediction(self):
        category = predict_category(
            "Wi-Fi is not working in Block C"
        )

        self.assertEqual(category, "IT & Wi-Fi")

    def test_hostel_prediction(self):
        category = predict_category(
            "There is no water supply in the hostel"
        )

        self.assertEqual(category, "Hostel")

    def test_high_priority(self):
        priority = predict_priority(
            "Emergency! There is no water supply"
        )

        self.assertEqual(priority, "High")