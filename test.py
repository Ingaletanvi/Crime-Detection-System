import unittest
from unittest.mock import patch, MagicMock
from app import send_email_alert, extract_frames, senet_detect_crime, login  # Replace `main_module` with your module's name
import streamlit as st
import os

class TestCrimeDetectionApp(unittest.TestCase):

    @patch('streamlit.file_uploader')
    @patch('streamlit.video')
    def test_file_upload(self, mock_video, mock_file_uploader):
        mock_file_uploader.return_value = open('test_video.mp4', 'rb')
        
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        st.video(uploaded_file)

        mock_file_uploader.assert_called_once()
        mock_video.assert_called_once_with(uploaded_file)

    @patch('cv2.VideoCapture')
    @patch('os.makedirs')
    def test_extract_frames(self, mock_makedirs, mock_videocapture):
        mock_capture = MagicMock()
        mock_videocapture.return_value = mock_capture
        mock_capture.isOpened.return_value = True
        mock_capture.read.side_effect = [(True, 'frame')] * 30 + [(False, None)]  # Simulate 30 frames

        output_dir = 'frames/temp/'
        video_path = 'test_video.mp4'
        extract_frames(video_path, output_dir)
        
        mock_makedirs.assert_called_once_with(output_dir)
        self.assertTrue(len(os.listdir(output_dir)) > 0)  # Check if frames were extracted

    @patch('tensorflow.keras.models.load_model')
    @patch('cv2.VideoCapture')
    def test_senet_model_prediction(self, mock_videocapture, mock_load_model):
        mock_capture = MagicMock()
        mock_videocapture.return_value = mock_capture
        mock_capture.isOpened.return_value = True
        mock_capture.read.side_effect = [(True, MagicMock())] * 30 + [(False, None)]  # Simulate frames

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_model.predict.return_value = [[0.8]]  # Simulating high confidence for crime

        video_path = 'test_video.mp4'
        is_crime, prediction = senet_detect_crime(video_path, mock_model)
        
        self.assertTrue(is_crime)
        self.assertEqual(prediction, 0.8)

    @patch('smtplib.SMTP')
    def test_email_alert_sent(self, mock_smtp):
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
        
        send_email_alert("Crime Alert!", "Crime detected in video")
        mock_smtp_instance.send_message.assert_called_once()

    @patch('streamlit.text_input')
    @patch('streamlit.button')
    def test_login(self, mock_button, mock_text_input):
        mock_text_input.side_effect = ['admin', 'admin123']
        mock_button.return_value = True
        
        login()
        
        self.assertTrue(st.session_state['logged_in'])
        self.assertEqual(st.session_state['username'], 'admin')

if __name__ == '__main__':
    unittest.main()
