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

Expected category: IT & Wi-Fi.

And:

```text
Emergency! There is no water supply in the hostel.
```

Expected category: Hostel and priority: High.

## Current Gaps

The prototype does not provide a custom admin dashboard, advanced multilingual support, or a more advanced complaint-intelligence pipeline. These are intentionally suitable areas for participant contributions.

## Main Technical Concepts

- Django request/response flow
- Django ORM and SQLite
- TF-IDF text representation
- Logistic Regression classification
- Cosine similarity
- HTML/CSS forms
- Automated Django tests
