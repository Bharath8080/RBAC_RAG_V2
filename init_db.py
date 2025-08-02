#!/usr/bin/env python3
"""
Initialize MongoDB database with sample users for FinSolve RBAC Chatbot
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

def init_database():
    """Initialize the database with sample users"""
    
    if not MONGO_URI or not DB_NAME:
        print("❌ Error: MONGO_URI and DB_NAME must be set in .env file")
        return
    
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        users_collection = db["users"]
        
        # Sample users data
        sample_users = [
            {
                "username": "Tony",
                "password": "password123",
                "role": "engineering",
                "email": "tony@finsolve.com",
                "department": "Engineering"
            },
            {
                "username": "Bruce",
                "password": "securepass",
                "role": "marketing",
                "email": "bruce@finsolve.com",
                "department": "Marketing"
            },
            {
                "username": "Sam",
                "password": "financepass",
                "role": "finance",
                "email": "sam@finsolve.com",
                "department": "Finance"
            },
            {
                "username": "Peter",
                "password": "pete123",
                "role": "engineering",
                "email": "peter@finsolve.com",
                "department": "Engineering"
            },
            {
                "username": "Sid",
                "password": "sidpass123",
                "role": "marketing",
                "email": "sid@finsolve.com",
                "department": "Marketing"
            },
            {
                "username": "Natasha",
                "password": "hrpass123",
                "role": "hr",
                "email": "natasha@finsolve.com",
                "department": "HR"
            }
        ]
        
        # Clear existing users
        users_collection.delete_many({})
        print("🗑️  Cleared existing users")
        
        # Insert sample users
        result = users_collection.insert_many(sample_users)
        print(f"✅ Inserted {len(result.inserted_ids)} users")
        
        # Display users
        print("\n📋 Sample Users:")
        print("=" * 50)
        for user in users_collection.find():
            print(f"👤 {user['username']} - {user['role'].capitalize()} ({user['email']})")
        
        print("\n🎉 Database initialized successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    print("🚀 Initializing FinSolve RBAC Database...")
    print("=" * 50)
    init_database() 