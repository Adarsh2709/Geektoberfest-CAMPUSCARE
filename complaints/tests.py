from django.test import TestCase, Client
from django.urls import reverse

from .models import Complaint
from .ml import (
    predict_category,
    predict_category_with_confidence,
    predict_priority,
    detect_language,
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
        self.assertEqual(complaint.language, "English")
        self.assertAlmostEqual(complaint.confidence, 0.92, places=2)

    def test_multilingual_complaint_creation(self):
        complaint = Complaint.objects.create(
            student_name="Rohit Sharma",
            description="हॉस्टल में पानी नहीं आ रहा है",
            category="Hostel",
            priority="High",
            department="Hostel Management",
            language="Hindi"
        )
        self.assertEqual(complaint.language, "Hindi")
        self.assertEqual(complaint.category, "Hostel")

    def test_complaint_default_confidence(self):
        complaint = Complaint.objects.create(
            student_name="Jane Doe",
            description="Hostel room tap is leaking"
        )
        self.assertEqual(complaint.confidence, 1.0)


class ComplaintLanguageDetectionTest(TestCase):

    def test_english_detection(self):
        self.assertEqual(
            detect_language("Wi-Fi is not working in Block C"),
            "English"
        )
        self.assertEqual(
            detect_language("Emergency! There is no water supply in hostel"),
            "English"
        )

    def test_hindi_detection(self):
        self.assertEqual(
            detect_language("हॉस्टल में पानी नहीं आ रहा है"),
            "Hindi"
        )
        self.assertEqual(
            detect_language("क्लासरूम का प्रोजेक्टर काम नहीं कर रहा है"),
            "Hindi"
        )

    def test_hinglish_detection(self):
        self.assertEqual(
            detect_language("Hostel mein paani nahi aa raha hai"),
            "Hinglish"
        )
        self.assertEqual(
            detect_language("Block C mein wifi kaam nahi kar raha"),
            "Hinglish"
        )

    def test_marathi_detection(self):
        self.assertEqual(
            detect_language("वसतिगृहात पाणी येत नाही आहे"),
            "Marathi"
        )
        self.assertEqual(
            detect_language("वर्गातील प्रोजेक्टर व्यवस्थित चालत नाही आहे"),
            "Marathi"
        )


class ComplaintMLCategoryTest(TestCase):

    # English Category Tests
    def test_english_it_prediction(self):
        self.assertEqual(
            predict_category("Wi-Fi is not working in Block C"),
            "IT & Wi-Fi"
        )

    def test_english_hostel_prediction(self):
        self.assertEqual(
            predict_category("There is no water supply in the hostel"),
            "Hostel"
        )

    def test_english_infrastructure_prediction(self):
        self.assertEqual(
            predict_category("Classroom projector is damaged in Main Building"),
            "Infrastructure"
        )

    def test_english_transport_prediction(self):
        self.assertEqual(
            predict_category("College bus is arriving late"),
            "Transport"
        )

    def test_english_academic_prediction(self):
        self.assertEqual(
            predict_category("Internal marks are missing on portal"),
            "Academic"
        )

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

    # Hindi Category Tests
    def test_hindi_it_prediction(self):
        self.assertEqual(
            predict_category("ब्लॉक सी में वाई-फाई काम नहीं कर रहा है"),
            "IT & Wi-Fi"
        )

    def test_hindi_hostel_prediction(self):
        self.assertEqual(
            predict_category("हॉस्टल में पानी की कोई सप्लाई नहीं आ रही है"),
            "Hostel"
        )

    def test_hindi_infrastructure_prediction(self):
        self.assertEqual(
            predict_category("क्लासरूम का प्रोजेक्टर काम नहीं कर रहा है"),
            "Infrastructure"
        )

    def test_hindi_transport_prediction(self):
        self.assertEqual(
            predict_category("कॉलेज बस बहुत देरी से आ रही है"),
            "Transport"
        )

    def test_hindi_academic_prediction(self):
        self.assertEqual(
            predict_category("पोर्टल पर मेरे मार्क्स नहीं दिख रहे हैं"),
            "Academic"
        )

    # Hinglish Category Tests
    def test_hinglish_it_prediction(self):
        self.assertEqual(
            predict_category("Block C mein wifi kaam nahi kar raha hai"),
            "IT & Wi-Fi"
        )

    def test_hinglish_hostel_prediction(self):
        self.assertEqual(
            predict_category("Hostel mein paani bilkul nahi aa raha hai"),
            "Hostel"
        )

    def test_hinglish_infrastructure_prediction(self):
        self.assertEqual(
            predict_category("Classroom projector kaam nahi kar raha hai"),
            "Infrastructure"
        )

    def test_hinglish_transport_prediction(self):
        self.assertEqual(
            predict_category("College bus bohot late aa rahi hai"),
            "Transport"
        )

    def test_hinglish_academic_prediction(self):
        self.assertEqual(
            predict_category("Portal par mere marks show nahi ho rahe hain"),
            "Academic"
        )

    # Marathi Category Tests
    def test_marathi_it_prediction(self):
        self.assertEqual(
            predict_category("ब्लॉक सी मध्ये वाय-फाय व्यवस्थित चालत नाही आहे"),
            "IT & Wi-Fi"
        )

    def test_marathi_hostel_prediction(self):
        self.assertEqual(
            predict_category("वसतिगृहात पाण्याचा पुरवठा बंद आहे"),
            "Hostel"
        )

    def test_marathi_infrastructure_prediction(self):
        self.assertEqual(
            predict_category("वर्गातील प्रोजेक्टर व्यवस्थित चालत नाही"),
            "Infrastructure"
        )

    def test_marathi_transport_prediction(self):
        self.assertEqual(
            predict_category("महाविद्यालयाची बस खूप उशिरा येत आहे"),
            "Transport"
        )

    def test_marathi_academic_prediction(self):
        self.assertEqual(
            predict_category("पोर्टलवर परीक्षेचे गुण दिसत नाहीत"),
            "Academic"
        )


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
    def test_english_priority(self):
        self.assertEqual(
            predict_priority("Emergency! There is no water supply"),
            "High"
        )
        self.assertEqual(
            predict_priority("General query about library hours"),
            "Low"
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

    def test_hindi_priority(self):
        self.assertEqual(
            predict_priority("आपातकालीन: हॉस्टल में पानी नहीं आ रहा"),
            "High"
        )
        self.assertEqual(
            predict_priority("क्लासरूम का पंखा काम नहीं कर रहा"),
            "Medium"
        )
        self.assertEqual(
            predict_priority("लाइब्रेरी के बारे में जानकारी चाहिए"),
            "Low"
        )

    def test_hinglish_priority(self):
        self.assertEqual(
            predict_priority("Khatra: hostel mein open wire hai"),
            "High"
        )
        self.assertEqual(
            predict_priority("Classroom ka projector kharab hai"),
            "Medium"
        )
        self.assertEqual(
            predict_priority("Timetable ki request"),
            "Low"
        )

    def test_marathi_priority(self):
        self.assertEqual(
            predict_priority("आणीबाणी: वसतिगृहात पाणी नाही"),
            "High"
        )
        self.assertEqual(
            predict_priority("वर्गातील पंखा बंद पडले आहे"),
            "Medium"
        )
        self.assertEqual(
            predict_priority("ग्रंथालयाची माहिती हवी आहे"),
            "Low"
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

    def test_duplicate_detection_multilingual(self):
        Complaint.objects.create(
            student_name="Student 1",
            description="हॉस्टल में पानी नहीं आ रहा है",
            category="Hostel",
            priority="High",
            department="Hostel Management",
            language="Hindi"
        )

        # Similar complaint in Hindi should be flagged
        is_dup = detect_duplicate("हॉस्टल में पानी नहीं आ रहा")
        self.assertTrue(is_dup)

        # Unrelated complaint should not be flagged
        is_not_dup = detect_duplicate("College bus is late")
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

    def test_home_page_post_hindi(self):
        response = self.client.post(
            reverse("home"),
            {
                "student_name": "Aarav Sharma",
                "description": "हॉस्टल में पानी की कोई सप्लाई नहीं आ रही है"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hostel")
        self.assertContains(response, "Hindi")

        complaint = Complaint.objects.last()
        self.assertEqual(complaint.student_name, "Aarav Sharma")
        self.assertEqual(complaint.category, "Hostel")
        self.assertEqual(complaint.department, "Hostel Management")
        self.assertEqual(complaint.language, "Hindi")

    def test_home_page_post_hinglish(self):
        response = self.client.post(
            reverse("home"),
            {
                "student_name": "Priya Verma",
                "description": "Block C mein wifi kaam nahi kar raha hai"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "IT &amp; Wi-Fi")
        self.assertContains(response, "Hinglish")

        complaint = Complaint.objects.last()
        self.assertEqual(complaint.language, "Hinglish")
        self.assertEqual(complaint.category, "IT & Wi-Fi")
        self.assertEqual(complaint.department, "IT Department")

    def test_home_page_post_marathi(self):
        response = self.client.post(
            reverse("home"),
            {
                "student_name": "Tanmay Patil",
                "description": "वसतिगृहात पाण्याचा पुरवठा बंद आहे"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hostel")
        self.assertContains(response, "Marathi")

        complaint = Complaint.objects.last()
        self.assertEqual(complaint.language, "Marathi")
        self.assertEqual(complaint.category, "Hostel")
