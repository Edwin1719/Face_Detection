# Face_Detection
![Logo](https://media.licdn.com/dms/image/v2/C4D12AQGX7iDfdCYdVw/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1608046439420?e=2147483647&v=beta&t=v4TuOM-dN4uvyOzwamgzkyQitV12HYmRqP5NZo1aRxE)

## This project implements a web app for facial recognition using Streamlit and DeepFace. Upload images, identify faces by comparing them with a directory, and analyze age, gender, race, and emotions. Features a user-friendly interface with result visualization and developer details.

## Features

- **Set Up the Application:** Configure the Streamlit app with a title, layout, and initial sidebar state. Add a logo at the top left using HTML and CSS.
- **Load and Display Image:** Implement a file uploader for users to upload images. Load and display the uploaded image in a format usable by OpenCV.
- **Face Identification and Analysis:** Compare the uploaded image with images in a specified directory using DeepFace to identify faces. Perform facial analysis to detect age, gender, race, and emotion, and display the results.
- **User Interface and Results Visualization:** Design a user-friendly interface with centered text and headers. Display the analysis results, including drawing rectangles around detected faces and showing detailed information.
- **Footer with Developer Info:** Add a footer with the developer’s contact information and social media icons.

## Technologies used

- **Streamlit:** Used to create the web application interface, handle user interactions, and display results.
- **OpenCV:** Utilized for image processing tasks, such as loading, converting, and displaying images.
- **DeepFace:** Employed for facial recognition and analysis, including identifying faces and detecting age, gender, race, and emotions.
- **NumPy:** Used for handling arrays and converting image data for processing with OpenCV.
- **os:** Utilized for file and directory operations, such as checking if a directory exists and iterating through files.
- **HTML and CSS:** Used to enhance the visual presentation of the app, including adding a logo and styling text.
- **st_social_media_links:** A Streamlit component used to add social media icons and links to the footer of the app.

## **Documentation**
! https://omes-va.com/reconocimiento-facial-python-opencv/
! https://www.youtube.com/watch?v=CPZvEgxKoJk
