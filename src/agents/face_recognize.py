import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient
import sys
from pathlib import Path

# ---------- CONFIG ----------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "FamilyGuard"
COLLECTION_NAME = "Members"
TOLERANCE = 0.5  # Lower = stricter matching
# ----------------------------

def get_database():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        return db[COLLECTION_NAME]
    except Exception as e:
        print(f"❌ MongoDB Connection Error: {e}")
        sys.exit(1)

def get_face_encodings(frame):
    """
    Converts BGR frame to RGB and returns encodings.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_frame, model="hog")
    encodings = face_recognition.face_encodings(rgb_frame, boxes)
    return encodings

def create_user_profile(doc_id, name, age, gender, allergies, city, state, email):
    """
    Helper to format the data exactly as requested.
    """
    return {
        "ID": str(doc_id),
        "Name": name,
        "Age": age,
        "Gender": gender,
        "Allergies": allergies,
        "City": city,
        "State": state,
        "Email": email  # <--- Added Email Field
    }

def face_recognice():
    collection = get_database()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible.")
        return

    print("👀 Scanning... (Look at the camera)")

    found_encoding = None
    
    # --- STEP 1: AUTO-DETECT LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Show the video feed so user knows they are in frame
        cv2.imshow("Auto-Scan (Press 'q' to quit)", frame)

        # check for faces
        encodings = get_face_encodings(frame)
        
        if encodings:
            # Face found! Grab the first one and stop the loop.
            found_encoding = encodings[0]
            print("✅ Face Detected! Processing...")
            break
        
        # Quit option
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if found_encoding is None:
        return

    # --- STEP 2: SEARCH DATABASE ---
    # Fetch all known faces
    known_encodings = []
    known_docs = []
    
    for doc in collection.find():
        if "encoding" in doc:
            known_encodings.append(np.array(doc["encoding"]))
            known_docs.append(doc)

    user_profile = {}

    if known_encodings:
        distances = face_recognition.face_distance(known_encodings, found_encoding)
        best_idx = np.argmin(distances)
        
        if distances[best_idx] <= TOLERANCE:
            # --- MATCH FOUND ---
            doc = known_docs[best_idx]
            user_profile = create_user_profile(
                doc_id=doc["_id"],
                name=doc.get("Name"),
                age=doc.get("Age"),
                gender=doc.get("Gender"),
                allergies=doc.get("Allergies"),
                city=doc.get("City"),
                state=doc.get("State"),
                email=doc.get("Email") # <--- Retrieve Email from DB
            )
            print("\n✅ Known Member Found.")
        else:
            print("\n❓ Unknown Face Detected.")
    else:
         print("\nℹ Database Empty. New User.")

    if not user_profile:
        print("\n--- ENTER MEMBER DETAILS ---")
        name = input("Name: ").strip()
        while True:
            try:
                age = int(input("Age: ").strip())
                break
            except ValueError:
                print("Age must be a number.")
        gender = input("Gender: ").strip()
        allergies = input("Allergies: ").strip()
        city = input("City: ").strip()
        state = input("State: ").strip()
        email = input("Email: ").strip() # <--- Ask for Email

        new_doc = {
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Allergies": allergies,
            "City": city,
            "State": state,
            "Email": email, # <--- Save Email to DB
            "encoding": found_encoding.tolist()
        }
        
        res = collection.insert_one(new_doc)
        
        user_profile = create_user_profile(
            doc_id=res.inserted_id,
            name=name,
            age=age,
            gender=gender,
            allergies=allergies,
            city=city,
            state=state,
            email=email # <--- Pass Email to return object
        )
        print("✅ New Member Saved.")

    print("\n⬇⬇⬇ RETURNED DATA ⬇⬇⬇")
    print(user_profile)
    return user_profile

def get_user_knowledge_profile():
    """
    Runs facial recognition to fetch or create user data,
    then stores that data in the knowledge/ directory under a unique subfolder
    and returns the full path to that file.
    """

    user_data = face_recognice()
    if not user_data:
        raise ValueError("No user recognized or created.")

    project_root = Path(__file__).resolve().parents[2]

    knowledge_root = project_root / "knowledge"
    user_dir = knowledge_root / f"user_{user_data['ID']}"
    user_dir.mkdir(parents=True, exist_ok=True)
    profile_file = user_dir / "profile.txt"

    profile_file.write_text(
        f"User Profile:\n"
        f"Name: {user_data['Name']}\n"
        f"Age: {user_data['Age']}\n"
        f"Gender: {user_data['Gender']}\n"
        f"Allergies: {user_data['Allergies']}\n"
        f"City: {user_data['City']}\n"
        f"State: {user_data['State']}\n"
        f"Email: {user_data['Email']}\n"
    )

    print(f"✅ User profile saved to {profile_file}")
    return user_data, profile_file