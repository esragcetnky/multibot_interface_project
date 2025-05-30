# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests
import os
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT,  "shared", "credentials.yml")

def load_credentials():
    """Load API credentials from YAML file."""
    with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
creds = load_credentials()
# Extract Tavily API key from credentials
TAVILY_API_KEY = creds.get("tavily_api_key", "")

def tavily_search(query, max_results=5):
    """
    Search the web using the Tavily API.
    Args:
        query (str): The search query.
        max_results (int): Number of results to return.
    Returns:
        list: List of search result dicts, or an empty list if search fails.
    """
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "max_results": max_results
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        # Log or handle error as needed
        print(f"Tavily search error: {e}")
        return []