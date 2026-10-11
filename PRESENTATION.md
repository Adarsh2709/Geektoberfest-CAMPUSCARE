# CAMPUSCARE — Presentation Guide

## Problem

Campus complaints are often submitted as unstructured text and can be difficult to categorize, prioritize and route manually.

## Proposed Solution

CAMPUSCARE provides a smart complaint intake workflow.

A student submits a complaint and the system:
1. Predicts a complaint category using TF-IDF and Logistic Regression.
2. Assigns a priority using text-based rules.
3. Routes the complaint to a department.
4. Checks whether a similar complaint already exists.
5. Stores the complaint in the database.

## Demo Flow

Use examples such as:

```text
Wi-Fi is not working in Block C
```
Expected language: English, category: IT & Wi-Fi.

```text
Emergency! There is no water supply in the hostel.
```
Expected language: English, category: Hostel, priority: High.

```text
हॉस्टल में पानी नहीं आ रहा है
```
Expected language: Hindi, category: Hostel.

```text
Hostel mein paani nahi aa raha hai
```
Expected language: Hinglish, category: Hostel.

```text
वसतिगृहात पाण्याचा पुरवठा बंद आहे
```
Expected language: Marathi, category: Hostel.

## Current Gaps

The prototype provides full multilingual complaint intake (English, Hindi, Hinglish, Marathi). Open areas for future contributions include building a staff admin analytics dashboard, speech-to-text input, and automated email/SMS status alerts.

## Main Technical Concepts

- Django request/response flow
- Django ORM and SQLite
- TF-IDF text representation
- Logistic Regression classification
- Cosine similarity
- HTML/CSS forms
- Automated Django tests
