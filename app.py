import helper
from math import sqrt, atan
import cv2
import time


class SlouchApp:
    def __init__(self, use_webcam=False):
        self.use_webcam = use_webcam
        self.start_detect_slouch = False
        self.eye_classifier = helper.get_eye_classifier()
        self.face_classifier = helper.get_face_classifier()
        self.distance_reference = helper.get_distance_reference()
        self.thoracolumbar_tolerance = helper.get_thoracolumbar_tolerance()

        # Initialize face distance and head tilt to 0.
        self.distance = 0
        self.angle = 0

        # Did user intend to use webcam?
        if not use_webcam:
            # No.
            # Use video passed in the parameter.
            args = helper.arguments()
            self.video = cv2.VideoCapture(args["video"])
        else:
            # Yes.
            # Set self.video to 0.
            self.video = cv2.VideoCapture(0)

    def open_camera_or_video(self):
        """
        Open the webcam or video.
        """
        # Keep showing video.
        while self.video.isOpened():
            # Read the video.
            frame_read_flag, bgr_image = self.video.read()

            # Are there any frames yet?
            if bgr_image is None:
                # No.
                # Stop video capture.
                break

            # Display quit option.
            cv2.putText(bgr_image, "Press Q to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Did user locked in their proper posture?
            if self.start_detect_slouch:
                # Yes.
                # Start detecting slouch.
                self.process_slouch_detection(bgr_image)
            else:
                # No.
                # Show controls for setting distance reference.
                cv2.putText(bgr_image, "Please sit properly then press C to lock in your proper sitting position.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display webcam/video output to window.
            cv2.imshow("Slouch Detection", bgr_image)

            # Did user pressed `q`?
            if cv2.waitKey(1) & 0xFF == ord("q"):
                # Yes.
                # Break out of loop to stop video capture.
                break
            elif cv2.waitKey(1) & 0xFF == ord("c"):
                detected_face_flag, face = self.detect_face(bgr_image)

                # Are there any faces detected?
                if detected_face_flag:
                    # Yes.
                    # Set distance reference.
                    self.distance_reference = self.determine_camera_face_distance(face)
                    print("Using user's current distance")
                else:
                    print("Using default distance reference")

                self.start_detect_slouch = True

        # Release the video capture and destroy all windows if user opted to stop the video capture.
        self.video.release()
        cv2.destroyAllWindows()

    def process_slouch_detection(self, bgr_image):
        # Detect face.
        detected_face_flag, face = self.detect_face(bgr_image)

        # Are there any face detected?
        if detected_face_flag:
            # Yes.
            # Determine distance of face from the camera.
            self.distance = self.determine_camera_face_distance(face)

            # Is user slouching?
            slouch_flag = self.is_user_slouching()

            if slouch_flag:
                cv2.putText(bgr_image, "You are slouching", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(bgr_image, "You are sitting properly", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Detect eyes to determine head angle or if head is tilted.
            # x, y, w, h = face
            # face_image = bgr_image[y:y+h, x:x+w]
            # head_tilt_flag, self.angle = self.determine_head_angle(face_image)

            # Was the head tilting properly determined?
            # if head_tilt_flag:
            #     # Yes.
            #     # Is user slouching?
            #     slouch_flag = self.is_user_slouching()
            #
            #     if slouch_flag:
            #         cv2.putText(bgr_image, "You are slouching", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     else:
            #         cv2.putText(bgr_image, "You are sitting properly", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # else:
            #     print(self.angle)

    def detect_face(self, bgr_image):
        """
        Detect face using face classifier.
        :param bgr_image: The video capture where the face will be detected.
        :return: The result whether the face is found and the face found or an error message.
        """
        # Load face classifier.
        face_classifier = cv2.CascadeClassifier(self.face_classifier)

        # Detect face in the image.
        faces = face_classifier.detectMultiScale(image=bgr_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)

        try:
            # Assume largest face is the subject.
            # Index 0 is largest face.
            face = faces[0]

            return True, face
        except IndexError:
            # No face found. Return false and error message.
            return False, "No faces found."

    def determine_camera_face_distance(self, face):
        """
        Determine the distance of the face from the camera.
        :param face: The detected face.
        :return:
        """
        # Unpack the face position (x, y) and dimensions (w, h) from found face.
        x, y, w, h = face

        # Calculate the distance.
        distance = sqrt(y**2 + w**2)

        return distance

    def determine_head_angle(self, face_image):
        """
        Detect eyes to determine the angle of the head tilting.
        :return: The angle of the head tilting.
        """
        # Load eye classifier.
        eye_classifier = cv2.CascadeClassifier(self.eye_classifier)

        # Detect eyes.
        eyes = eye_classifier.detectMultiScale(face_image)

        # Are there any eyes detected and is it more than 1 eye detected?
        if len(eyes) > 1:
            # Yes.
            # Read only the first 2 eyes detected.
            left_eye = eyes[0]
            right_eye = eyes[1]

            slope = (left_eye[1] - right_eye[1] / left_eye[0] - right_eye[0])
            angle = abs(atan(slope))

            return True, angle
        else:
            return False, "No eyes detected."

    def is_user_slouching(self):
        """
        Check whether user is slouching.
        :return: Boolean value to determine slouching.
        """
        c_min = self.distance_reference * (1.0 - self.thoracolumbar_tolerance)
        c_max = self.distance_reference * (1.0 + self.thoracolumbar_tolerance)

        slouching_flag = True
        # head_tilt_flag = False

        # Is user slouching?
        if c_min <= self.distance <= c_max:
            # No.
            slouching_flag = False

        # Is user's head tilted?
        # if self.angle > 0.4:
        #     # Yes.
        #     head_tilt_flag = True

        return slouching_flag
