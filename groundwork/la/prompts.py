"""Prompt templates for the LocateAnything auto-labeler.

Templates are verbatim from the LocateAnything-3B model card. Different
*descriptions* change what the teacher detects, so we expose a helper that
slots any phrase into the multi-instance detection template — the description
usually comes from the project's class names.
"""
from __future__ import annotations

# Verbatim templates from the model card ("Supported Tasks & Prompt Templates").
DETECT_MULTI = "Locate all the instances that match the following description: {desc}."
DETECT_CATEGORIES = "Locate all the instances that matches the following description: {cats}."
GROUND_SINGLE = "Locate a single instance that matches the following description: {desc}."
POINT = "Point to: {desc}."


def detect_prompt(desc: str) -> str:
    """Multi-instance detection prompt for a free-form description."""
    return DETECT_MULTI.format(desc=desc)


def categories_prompt(categories: list[str]) -> str:
    """Detection prompt for a list of categories (joined with </c>)."""
    return DETECT_CATEGORIES.format(cats="</c>".join(categories))


def point_prompt(desc: str) -> str:
    return POINT.format(desc=desc)
