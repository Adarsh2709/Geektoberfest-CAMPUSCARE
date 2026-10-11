from django.test import TestCase, Client
from django.urls import reverse

from .models import Complaint
from .ml import (
    predict_category,
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
            department="IT Department"
        )

        self.assertEqual(complaint.student_name, "Test Student")
        self.assertEqual(complaint.category, "IT & Wi-Fi")
        self.assertEqual(complaint.priority, "Medium")
        self.assertEqual(complaint.language, "English")

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


class ComplaintMLPriorityTest(TestCase):

    def test_english_priority(self):
        self.assertEqual(
            predict_priority("Emergency! There is no water supply"),
            "High"
        )
        self.assertEqual(
            predict_priority("Wi-Fi is not working in Block A"),
            "Medium"
        )
        self.assertEqual(
            predict_priority("General query about library hours"),
            "Low"
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


class ComplaintDuplicateDetectionTest(TestCase):

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


class ComplaintViewEndToEndTest(TestCase):

    def setUp(self):
        self.client = Client()

    def test_home_page_get(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "CampusCare")

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