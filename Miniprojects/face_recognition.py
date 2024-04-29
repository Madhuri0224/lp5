import cv2

def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

# Load the pre-trained cascade classifier for face detection
face_cap = cv2.CascadeClassifier("C:/Users/MYPC/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# Initialize the video capture object
video_cap = cv2.VideoCapture(0)

# Option to choose between live camera feed and image upload
mode = input("Enter 'live' for live camera feed or 'image' to upload an image: ")

if mode.lower() == 'live':
    while True:
        # Read a frame from the video capture
        ret, video_data = video_cap.read()

        # Check if the frame is successfully read
        if not ret:
            print("Failed to read frame from the camera.")
            break

        # Detect faces in the frame
        video_data = detect_faces(video_data)

        # Display the frame with detected faces
        cv2.imshow("Faces Detected", video_data)

        # Check for key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' key or Esc key to exit
            break

elif mode.lower() == 'image':
    # Path to the image file
    image_path = input("Enter the path to the image file: ")

    # Read the image
    image = cv2.imread(image_path)

    # Resize the image to fit the window
    height, width = image.shape[:2]
    max_height = 600
    max_width = 800
    if height > max_height or width > max_width:
        scale = min(max_height/height, max_width/width)
        image = cv2.resize(image, (int(width*scale), int(height*scale)))

    # Detect faces in the image
    result_image = detect_faces(image)

    # Display the result
    cv2.imshow("Faces Detected", result_image)
    cv2.waitKey(0)

else:
    print("Invalid mode entered. Please enter 'live' or 'image'.")

# Release the video capture object and close all windows
video_cap.release()
cv2.destroyAllWindows()

# C:/Users/MYPC/Downloads/LP5-minipr/img.jpg
