from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import tkinter as tk
import csv
import easygui
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import os
import pyttsx3
import dlib
engine = pyttsx3.init()
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)
def contact():
    mess._show(title='Contact us', message="Please contact us on : 'bhanuganesh342@gmail.com' ")

def check_haarcascadefile():
    exists = os.path.isfile(r"data_files\haarcascade_frontalface_default.xml")
    if not exists:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()

def save_pass():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            mess.show(title='Password Registered', message='New password was registered successfully!!')
            return
    op = (old.get())
    newp = (new.get())
    nnewp = (nnew.get())
    if (op == key):
        if (newp == nnewp):
            txf = open("TrainingImageLabel\psd.txt", "w")
            txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()

def change_pass():
    global master
    master = tk.Tk()
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")
    lbl4 = tk.Label(master, text='    Enter Old Password', bg='white', font=('comic', 12, ' bold '))
    lbl4.place(x=10, y=10)
    global old
    old = tk.Entry(master, width=25, fg="black", relief='solid', font=('comic', 12, ' bold '), show='*')
    old.place(x=180, y=10)
    lbl5 = tk.Label(master, text='   Enter New Password', bg='white', font=('comic', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black", relief='solid', font=('comic', 12, ' bold '), show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Password', bg='white', font=('comic', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid', font=('comic', 12, ' bold '), show='*')
    nnew.place(x=180, y=80)
    cancel = tk.Button(master, text="Cancel", command=master.destroy, fg="black", bg="red", height=1, width=25,
                       activebackground="white", font=('comic', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#00fcca", height=1, width=25,
                      activebackground="white", font=('comic', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()

def psw():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if password == key:
        TrainImages()
    elif password == None:
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered the wrong password')

def clear():
    txt.delete(0, 'end')
    res = "1)Train Images  >>>  2)Save Profile"
    message1.configure(text=res)

def clear2():
    txt2.delete(0, 'end')
    res = "1)Train Images  >>>  2)Save Profile"
    message1.configure(text=res)

def is_duplicate_id(student_id):
    exists = False
    with open("StudentDetails\StudentDetails.csv", 'r') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)  # Skip header row
        for row in reader:
            # Check if the row has at least 3 columns before accessing index 2
            if len(row) >= 3:
                existing_id = row[2]
                if existing_id == student_id:
                    exists = True
                    break
    return exists


output_folder = "emotion"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load pre-trained models for gender prediction
gender_net = cv2.dnn.readNetFromCaffe(
    "age_gender_models/deploy_gender.prototxt",
    "age_gender_models/gender_net.caffemodel")

# Mean values for gender prediction
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Male', 'Female']

# Load the pre-trained model for face detection
face_detector = dlib.get_frontal_face_detector()

# Load the pre-trained model for facial landmarks detection
landmark_predictor = dlib.shape_predictor("data_files\shape_predictor_68_face_landmarks.dat")

# Load the pre-trained model for emotion recognition
emotion_classifier = load_model(r'data_files\model.h5')

# Load the Haar cascade for eye pair detection
eye_cascade = cv2.CascadeClassifier(r"data_files\haarcascade_mcs_eyepair_big.xml")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to predict gender
def predict_gender(face_img):
    # Convert grayscale image to RGB
    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)

    # Preprocess the face image
    blob = cv2.dnn.blobFromImage(rgb_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Set input and perform forward pass
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()

    # Get predicted gender
    gender = gender_list[gender_preds[0].argmax()]

    return gender

def process_images(folder_path, output_folder):
    # Output directory for images with detected eye pairs
    eye_output_folder = os.path.join(output_folder, "output_detected_floder_img/")
    os.makedirs(eye_output_folder, exist_ok=True)

    # Check if the directory exists and is not empty
    if os.path.exists(folder_path) and os.listdir(folder_path):
        # Iterate over images in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Read the image
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = face_detector(gray)

                # Process each detected face
                for face in faces:
                    # Detect facial landmarks
                    landmarks = landmark_predictor(gray, face)
                    landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

                    # Extract the region of interest (ROI) containing the face
                    (x, y, w, h) = cv2.boundingRect(np.array([landmarks]))
                    roi_gray = gray[y:y + h, x:x + w]

                    # Check if the ROI is not empty before resizing
                    if not roi_gray.size == 0:
                        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                        # Preprocess the face image for emotion detection
                        roi_emotion = roi_gray.astype('float') / 255.0
                        roi_emotion = np.expand_dims(roi_emotion, axis=0)
                        roi_emotion = np.expand_dims(roi_emotion, axis=-1)

                        # Predict the emotion
                        prediction_emotion = emotion_classifier.predict(roi_emotion)[0]
                        label_emotion = emotion_labels[prediction_emotion.argmax()]

                        # Preprocess the face image for gender detection
                        roi_gender = cv2.resize(roi_gray, (227, 227))
                        gender = predict_gender(roi_gender)

                        # Draw a rectangle around the face and display the predicted emotion and gender labels
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(img, f'Emotion: {label_emotion}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(img, f'Gender: {gender}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Draw facial landmarks
                        for (x, y) in landmarks:
                            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

                        # Detect eyes in the face region
                        eyes = eye_cascade.detectMultiScale(roi_gray)

                        # Draw rectangles around detected eyes
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

                # Display the result
                cv2.imshow('Result', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Save processed image to output directory
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img)

                # Save image with detected eye pairs to separate output directory
                eye_output_path = os.path.join(eye_output_folder, filename)
                cv2.imwrite(eye_output_path, img)

        # Notify user that processing is complete

        print("Processing complete. Check the output folder for the processed images and the folder with eye pairs.")
        speak_text = f"HELLO ,BUDDY  EMOTION DETECTION IS COMPLETED THANK YOU  "

        # Call the text-to-speech engine to speak the message
        engine.say(speak_text)
        engine.runAndWait()
    else:
        print("Directory is empty or does not exist.")

def process_video():
    cap = cv2.VideoCapture(0)

    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    image_count = 0

    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector(gray)
        for face in faces:
            landmarks = landmark_predictor(gray, face)
            landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

            (x, y, w, h) = cv2.boundingRect(np.array([landmarks]))
            roi_gray = gray[y:y + h, x:x + w]

            if not roi_gray.size == 0:
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                roi_emotion = roi_gray.astype('float') / 255.0
                roi_emotion = np.expand_dims(roi_emotion, axis=0)
                roi_emotion = np.expand_dims(roi_emotion, axis=-1)

                prediction_emotion = emotion_classifier.predict(roi_emotion)[0]
                label_emotion = emotion_labels[prediction_emotion.argmax()]

                roi_gender = cv2.resize(roi_gray, (227, 227))
                gender = predict_gender(roi_gender)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f'Emotion: {label_emotion}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.putText(frame, f'Gender: {gender}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for (x_lm, y_lm) in landmarks:
                    cv2.circle(frame, (x_lm, y_lm), 1, (0, 0, 255), -1)

                file_name = os.path.join(output_folder, f"face_{image_count}.jpg")
                cv2.imwrite(file_name, frame[y:y + h, x:x + w])
                cv2.putText(frame, f'Emotion: {label_emotion}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

                with open(file_name.replace(".jpg", ".txt"), "w") as text_file:
                    text_file.write(label_emotion)

                image_count += 1

        cv2.imshow('Frame', frame)

        end_time = cv2.getTickCount()

        elapsed_time = (end_time - start_time) / cv2.getTickFrequency()

        if elapsed_time >= 50 or image_count >= 50:
            speak_text = f"HELLO ,BUDDY  EMOTION DETECTION IS COMPLETED THANK YOU  "

            # Call the text-to-speech engine to speak the message
            engine.say(speak_text)
            engine.runAndWait()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak_text = f"HELLO ,BUDDY  EMOTION DETECTION IS COMPLETED THANK YOU  "

            # Call the text-to-speech engine to speak the message
            engine.say(speak_text)
            engine.runAndWait()
            break

    cap.release()
    cv2.destroyAllWindows()

def process_video_or_images():
    msg = "Welcome to the Video/Image Processing System!"
    title = "Processing Options"
    choices = ["Process images from folder", "Process live video from webcam"]
    choice = easygui.buttonbox(msg, title, choices=choices)
    speak_text = f"HELLO ,YOUR CHOICES IS {choice}  "

    # Call the text-to-speech engine to speak the message
    engine.say(speak_text)
    engine.runAndWait()
    if choice == "Process images from folder":
        folder_path = easygui.diropenbox("Select the folder containing images:")
        output_folder = easygui.diropenbox("Select the output folder for processed images:")
        process_images(folder_path, output_folder)
    elif choice == "Process live video from webcam":
        process_video()
def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    assure_path_exists("EyeImages/")

    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")

    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()

    Id = (txt.get())
    name = (txt2.get())

    if not Id or not name:
        res = "Error: Please fill both ID and Name fields."
        message.configure(text=res)
        return

    if is_duplicate_id(Id):
        res = "Error: ID already exists. Please use a different ID."
        message.configure(text=res)
        return

    harcascadePath = r"data_files\haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    landmark_predictor = dlib.shape_predictor(r"data_files\shape_predictor_68_face_landmarks.dat")

    if ((name.isalpha()) or (' ' in name)):
        cam = cv2.VideoCapture(0)
        eye_cascade = cv2.CascadeClassifier(r"data_files\haarcascade_mcs_eyepair_big.xml")

        sampleNum = 0
        emotion = None
        try:
            while True:
                ret, img = cam.read()
                if not ret:
                    print("Error: Failed to capture frame from the camera.")
                    break

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    landmarks = landmark_predictor(gray, dlib.rectangle(x, y, x + w, y + h))

                    for landmark in landmarks.parts():
                        cv2.circle(img, (landmark.x, landmark.y), 1, (0, 0, 255), -1)

                    roi_gray = gray[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)

                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                        cv2.imwrite("EyeImages\ " + name + "." + str(serial) + "." + Id + '.' + str(
                            sampleNum) + "_eye.jpg", roi_gray[ey:ey + eh, ex:ex + ew])

                    sampleNum = sampleNum + 1
                    cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + Id + '.' + str(
                        sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    cv2.imshow('Taking Images', img)

                    if sampleNum == 100:
                        cam.release()
                        cv2.destroyAllWindows()
                        speak_text = f"HELLO ,BUDDY PLEASE WAIT  SHIFTING TO EMOTION DETECTION"

                        # Call the text-to-speech engine to speak the message
                        engine.say(speak_text)
                        engine.runAndWait()
                        emotion_code = process_video_or_images()

                        break

                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif sampleNum > 150:
                    break

        except Exception as e:
            print("An error occurred:", e)
        finally:
            cam.release()
            cv2.destroyAllWindows()

        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        if (name.isalpha() == False):
            res = "Error: Enter Correct name"
            message.configure(text=res)

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = r"data_files\haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Profile Saved Successfully"
    message1.configure(text=res)
    message.configure(text='Total Registrations till now  : ' + str(ID[0]))


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids
def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    for k in tv.get_children():
        tv.delete(k)
    msg = ''
    i = 0
    j = 0
    recognizer = cv2.face.LBPHFaceRecognizer_create() if cv2.__version__.startswith(
        '3') else cv2.face.LBPHFaceRecognizer_create()
    exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
    if exists3:
        recognizer.read("TrainingImageLabel\Trainner.yml")
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return
    harcascadePath = "data_files\haarcascade_frontalface_default.xml"
    eye_cascade = cv2.CascadeClassifier(r"data_files\haarcascade_mcs_eyepair_big.xml")
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    attendance = []  # Initialize the attendance list
    df = pd.read_csv("StudentDetails\StudentDetails.csv")  # Read the DataFrame here

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.5, 10)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)

            # Region of Interest (ROI) for eyes
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(im, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]

                speak_text = f"{bb},WELCOME TO THE OFFICE,THANKS FOR COMING"

                engine.say(speak_text)
                engine.runAndWait()
            else:
                Id = 'Unknown'
                bb = str(Id)

            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Taking Attendance', im)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    exists = os.path.isfile("Attendance\Attendance_" + date + ".csv")
    if exists:
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()
    else:
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvFile1.close()
    with open("Attendance\Attendance_" + date + ".csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if i > 1 and len(lines) >= 7:
                if i % 2 != 0:
                    iidd = str(lines[0]) + '   '
                    tv.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
    csvFile1.close()

    cam.release()
    cv2.destroyAllWindows()

global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {'01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
        }

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("Facial Recognition System")

image_path = r"background_images\background.jpeg"
try:
    image = Image.open(image_path)
    background_image = ImageTk.PhotoImage(image)
except Exception as e:
    print(f"Error loading image: {e}")
    background_image = None

if background_image:
    background_label = tk.Label(window, image=background_image)
    background_label.place(relwidth=1, relheight=1)

watermark_text = "**𝐵𝐻𝒜𝒩𝒰 𝒢𝒜𝒩𝐸𝒮𝐻✍**"

watermark_label = tk.Label(window, text=watermark_text, fg="white", bg="#FFA500",
                           font=('comic', 12, 'italic bold'))
watermark_label.pack(side=tk.BOTTOM)
# Watermark text
watermark_text = "**BHANU**"

watermark_label = tk.Label(window, text=watermark_text, fg="white", bg="#FFA500",
                           font=('comic', 12, 'italic bold'))

def update_watermark_position(event):
    window_width = event.width
    window_height = event.height
    watermark_label.place(x=window_width - 200, y=window_height - 30)

frame1 = tk.Frame(window, bg="#00FFFF")
frame1.place(relx=0.02, rely=0.17, relwidth=0.39, relheight=0.80)

image_path1 = r"background_images\frame1.jpg"
try:
    image = Image.open(image_path1)
    background_image1 = ImageTk.PhotoImage(image)
except Exception as e:
    print(f"Error loading image: {e}")
    background_image = None

# Create a label with the image
if background_image:
    background_label = tk.Label(frame1, image=background_image1)
    background_label.place(relwidth=1, relheight=1)

frame2 = tk.Frame(window, bg="#00FFFF")
frame2.place(relx=0.6, rely=0.17, relwidth=0.39, relheight=0.80)


image_path2 = r"background_images\frame2.jpg"
try:
    image = Image.open(image_path2)
    background_image2 = ImageTk.PhotoImage(image)
except Exception as e:
    print(f"Error loading image: {e}")
    background_image = None

if background_image:
    background_label = tk.Label(frame2, image=background_image2)
    background_label.place(relwidth=1, relheight=1)

message3 = tk.Label(window, text="Face Recognition  System", fg="white", bg="#FFA500",
                    width=55, height=1, font=('comic', 29, ' bold '))
message3.place(x=10, y=10)

frame3 = tk.Frame(window, bg="#ff61e5")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4,text=" " + day + "-" + mont[month] +" | ", fg="#ff61e5", bg="#2d420a", width=55, height=1,
                 font=('comic', 22, ' bold '))
datef.pack(fill='both', expand=1)

clock = tk.Label(frame3, fg="#ff61e5", bg="#2d420a", width=55, height=1, font=('comic', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

head2 = tk.Label(frame2, text="                 New Registrations                        ", fg="black", bg="#00FFFF",
                 font=('Georgia', 20, 'bold'))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1, text="                            Registered                                      ",
                 fg="black", bg="#00FFFF", font=('Georgia', 20, 'bold'))
head1.place(x=0, y=0)

lbl = tk.Label(frame2, text="Enter ID", width=20, height=1, fg="black", bg="#c79cff", font=('comic', 17, ' bold '))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=('comic', 15, ' bold '))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", width=20, fg="black", bg="#c79cff", font=('comic', 17, ' bold '))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=('comic', 15, ' bold '))
txt2.place(x=30, y=173)

message1 = tk.Label(frame2, text="1)Train Images  >>>  2)Save Profile", bg="#c79cff", fg="black", width=39, height=1,
                    activebackground="#3ffc00", font=('comic', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frame2, text="", bg="#c79cff", fg="black", width=39, height=1, activebackground="#3ffc00",
                   font=('comic', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frame1, text="DATA", width=20, fg="black", bg="#c79cff", height=1, font=('comic', 17, ' bold '))
lbl3.place(x=100, y=115)

res = 0
exists = os.path.isfile("StudentDetails\StudentDetails.csv")
if exists:
    with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text='Total Registrations till now  : ' + str(res))

menubar = tk.Menu(window, relief='ridge')
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Help', font=('comic', 29, ' bold '), menu=filemenu)

tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

clearButton = tk.Button(frame2, text="Clear", command=clear, fg="black", bg="#ff7221", width=11,
                        activebackground="white", font=('comic', 11, ' bold '))
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="black", bg="#ff7221", width=11,
                         activebackground="white", font=('comic', 11, ' bold '))
clearButton2.place(x=335, y=172)
takeImg = tk.Button(frame2, text="APPLY_CNN", command=TakeImages, fg="white", bg="#6d00fc", width=34, height=1,
                    activebackground="white", font=('comic', 15, ' bold '))
takeImg.place(x=30, y=300)
trainImg = tk.Button(frame2, text="RECORD_EMP", command=psw, fg="white", bg="#6d00fc", width=34, height=1,
                     activebackground="white", font=('comic', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frame1, text="SCAN THE FACE", command=TrackImages, fg="black", bg="#3ffc00", width=35, height=1,
                     activebackground="white", font=('comic', 15, ' bold '))
trackImg.place(x=30, y=50)
quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="#eb4600", width=35, height=1,
                       activebackground="white", font=('comic', 15, ' bold '))
quitWindow.place(x=30, y=450)

def welcome_message():
    engine = pyttsx3.init()
    engine.say("Welcome to the Facial Recognition System! We're glad to have you on board")
    engine.runAndWait()
window.configure(menu=menubar)
welcome_message()
window.mainloop()
# After your main code
engine.say("Thank you for choosing our Facial Recognition System! Your safety matters to us.")
engine.runAndWait()