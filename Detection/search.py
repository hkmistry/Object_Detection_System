# search.py
from flask import Blueprint, request, jsonify
import json
import os

search_bp = Blueprint('search', __name__)

# Example in-memory knowledge base
OBJECT_DATABASE = {
    "laptop": {"category": "Electronics", "confidence": 98, "description": "A portable computer"},
    "cup": {"category": "Household", "confidence": 95, "description": "Used for drinking liquids"},
    "chair": {"category": "Furniture", "confidence": 92, "description": "Seat for one person"},
    "book": {"category": "Stationery", "confidence": 85, "description": "A written or printed work"},
    "bottle": {"category": "Household", "confidence": 89, "description": "Container for liquids"}
}

@search_bp.route('/search_object', methods=['POST'])
def search_object():
    try:
        data = request.get_json()
        query = data.get("query", "").strip().lower()

        if not query:
            return jsonify({"success": False, "error": "Empty query"}), 400

        # Simple object match (you can replace this with AI model logic)
        results = []
        for name, info in OBJECT_DATABASE.items():
            if query in name or name in query:
                results.append({
                    "name": name.capitalize(),
                    "category": info["category"],
                    "confidence": info["confidence"],
                    "description": info["description"]
                })

        if not results:
            return jsonify({"success": True, "objects": []})

        return jsonify({"success": True, "objects": results})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
