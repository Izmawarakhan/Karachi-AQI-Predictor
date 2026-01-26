import os
from pymongo import MongoClient

class MongoDBFeatureStore:
    def __init__(self):
        # 1. Sabse pehle GitHub Action ka secret check karein
        mongo_uri = os.environ.get('MONGO_URI')
        
        # 2. Agar secret mil gaya (Cloud par), toh use karein
        if mongo_uri:
            print("🚀 Successfully connected to MongoDB Atlas Cloud!")
            self.client = MongoClient(mongo_uri)
        # 3. Agar nahi mila (Aapke apne computer par), toh localhost use karein
        else:
            print("🏠 MONGO_URI not found in environment. Connecting to Localhost...")
            self.client = MongoClient('mongodb://localhost:27017/')
            
        self.db = self.client.aqi_predictor

# Singleton instance
feature_store = MongoDBFeatureStore()