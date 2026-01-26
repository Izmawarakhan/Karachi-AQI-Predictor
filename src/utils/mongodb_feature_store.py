from pymongo import MongoClient
import os

class MongoDBFeatureStore:
    def __init__(self):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client.aqi_predictor

feature_store = MongoDBFeatureStore()