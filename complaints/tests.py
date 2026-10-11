from django.test import TestCase, Client
from django.urls import reverse

from .models import Complaint
from .ml import (
    predict_category,
    predict_category_with_confidence,
    predict_priority,
    detect_duplicate
)


class ComplaintModelTest(TestCase):

    def test_complaint_creation(self):
        complaint = Complaint.objects.create(
            student_name="Test Student",
            description="Wi-Fi is not working in Block A",
            category="IT & Wi-Fi",
            priority="Medium",
            department="IT Department",
            confidence=0.92
        )

        self.assertEqual(complaint.student_name, "Test Student")
        self.assertEqual(complaint.category, "IT & Wi-Fi")
        self.assertEqual(complaint.priority, "Medium")
        self.assertAlmostEqual(complaint.confidence, 0.92, places=2)

    def test_complaint_default_confidence(self):
        complaint = Complaint.objects.create(
            student_name="Jane Doe",
            description="Hostel room tap is leaking"
        )
        self.assertEqual(complaint.confidence, 1.0)


class ComplaintMLCategoryTest(TestCase):

    def test_it_wifi_prediction(self):
        category = predict_category("Wi-Fi is not working in Block C")
        self.assertEqual(category, "IT & Wi-Fi")

    def test_hostel_prediction(self):
        category = predict_category("There is no water supply in the hostel")
        self.assertEqual(category, "Hostel")

    def test_academic_prediction(self):
        category = predict_category("Internal marks are missing on student dashboard")
        self.assertEqual(category, "Academic")

    def test_infrastructure_prediction(self):
        category = predict_category("Classroom projector bulb is fused in Main Building")
        self.assertEqual(category, "Infrastructure")

    def test_transport_prediction(self):
        category = predict_category("College bus is arriving late every morning")
        self.assertEqual(category, "Transport")

    def test_eduroam_laptop_prediction(self):
        category = predict_category("Cannot connect to eduroam on my laptop in central library")
        self.assertEqual(category, "IT & Wi-Fi")

    def test_geyser_laundry_prediction(self):
        category = predict_category("Hostel geyser not heating water in 2nd floor bathroom")
        self.assertEqual(category, "Hostel")

    def test_air_conditioner_leak_prediction(self):
        category = predict_category("Air conditioner in seminar hall is leaking water and malfunctioning")
        self.assertEqual(category, "Infrastructure")

    def test_lecture_attendance_prediction(self):
        category = predict_category("Professor cancelled lecture without notice and attendance marked absent")
        self.assertEqual(category, "Academic")

    def test_shuttle_van_delay_prediction(self):
        category = predict_category("College shuttle van from metro station delayed by 45 minutes")
        self.assertEqual(category, "Transport")


class ComplaintMLConfidenceTest(TestCase):

    def test_confidence_return_structure(self):
        category, confidence = predict_category_with_confidence("Wi-Fi is down in Block C")
        self.assertEqual(category, "IT & Wi-Fi")
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_high_confidence_on_clear_samples(self):
        _, confidence = predict_category_with_confidence("College bus is arriving late every day")
        self.assertGreater(confidence, 0.50)


class ComplaintMLPriorityTest(TestCase):

    # High Priority Cases
    def test_emergency_priority(self):
        self.assertEqual(
            predict_priority("Emergency! There is no water supply"),
            "High"
        )

    def test_electric_shock_high_priority(self):
        self.assertEqual(
            predict_priority("Electric shock felt from metal switchboard in physics laboratory"),
            "High"
        )

    def test_food_poisoning_high_priority(self):
        self.assertEqual(
            predict_priority("Hostel mess food quality is spoiled and caused students vomiting"),
            "High"
        )

    def test_ceiling_collapse_high_priority(self):
        self.assertEqual(
            predict_priority("Ceiling plaster fell down on student desk in classroom 101"),
            "High"
        )

    def test_hall_ticket_deadline_high_priority(self):
        self.assertEqual(
            predict_priority("Hall ticket for upcoming end-semester exam is not generated"),
            "High"
        )

    # Medium Priority Cases
    def test_wifi_not_working_medium_priority(self):
        self.assertEqual(
            predict_priority("Wi-Fi is not working in Block A"),
            "Medium"
        )

    def test_projector_fused_medium_priority(self):
        self.assertEqual(
            predict_priority("Classroom projector bulb is fused in main block"),
            "Medium"
        )

    def test_tap_dripping_medium_priority(self):
        self.assertEqual(
            predict_priority("Bathroom tap is continuously dripping and wasting water"),
            "Medium"
        )

    def test_course_slides_missing_medium_priority(self):
        self.assertEqual(
            predict_priority("Course syllabus and lecture slides not uploaded on learning portal"),
            "Medium"
        )

    # Low Priority & Negation Cases
    def test_general_inquiry_low_priority(self):
        self.assertEqual(
            predict_priority("Inquiry regarding bus routes for the upcoming semester"),
            "Low"
        )

    def test_negation_not_an_emergency_low_priority(self):
        self.assertEqual(
            predict_priority("This is not an emergency, just need information on bus fee payment"),
            "Low"
        )


class ComplaintDuplicateDetectionTest(TestCase):

    def test_duplicate_detection(self):
        Complaint.objects.create(
            student_name="Student 1",
            description="Wi-Fi is not working in Block C",
            category="IT & Wi-Fi",
            priority="Medium",
            department="IT Department"
        )

        # Very similar complaint should be flagged
        is_dup = detect_duplicate("Wi-Fi is not working in Block C")
        self.assertTrue(is_dup)

        # Dissimilar complaint should not be flagged
        is_not_dup = detect_duplicate("Hostel food quality is terrible")
        self.assertFalse(is_not_dup)


class ComplaintViewTest(TestCase):

    def setUp(self):
        self.client = Client()

    def test_home_page_get(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "CampusCare")

    def test_home_page_post_creates_complaint_with_confidence(self):
        response = self.client.post(
            reverse("home"),
            {
                "student_name": "Rohan Gupta",
                "description": "College bus is arriving late every morning"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Transport")
        self.assertContains(response, "Confidence")

        complaint = Complaint.objects.last()
        self.assertEqual(complaint.student_name, "Rohan Gupta")
        self.assertEqual(complaint.category, "Transport")
        self.assertEqual(complaint.department, "Transport Office")
        self.assertGreater(complaint.confidence, 0.0)