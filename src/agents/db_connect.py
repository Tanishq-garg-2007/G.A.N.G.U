from pymongo import MongoClient

MONGO_URI = "mongodb+srv://ajaykathar30:EEk9w6fsnyYYLDGo@cluster0.441o7q2.mongodb.net/"

def get_database():
    try:
        client = MongoClient(MONGO_URI)
        db = client["medical_db"]   
        print("✅ MongoDB Connected Successfully!")
        return db
    except Exception as e:
        print("❌ Connection Error:", e)

if __name__ == "__main__":
    db = get_database()
    print("Current DB Name:", db.name)